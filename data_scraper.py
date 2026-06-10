import os
import glob
import json
import csv
import time
import datetime
import asyncio
from io import BytesIO

import aiohttp
import numpy as np
from PIL import Image

json_directory = r'/data/geolocation/panos_supplement/json'
output_directory = r'/data/geolocation/panos_supplement/images'
requested_ids_file = os.path.join(output_directory, 'requested_ids.json')
log_file = os.path.join(output_directory, 'panorama_log.csv')
request_limit = 300_000
tile_url = (
    "https://streetviewpixels-pa.googleapis.com/v1/tile"
    "?cb_client=maps_sv.tactile&panoid={pano_id}&x={x}&y={y}&zoom={zoom}&nbt=1&fover=2"
)

# Browser-style headers — cbk0 has been 403'ing without them and
# streetviewpixels-pa wants origin/referer set to look like a Maps embed call.
# Session-specific bits (x-browser-validation, x-client-data) are intentionally
# omitted; they're HMACs we can't reproduce and the endpoint serves fine without.
TILE_HEADERS = {
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.google.com",
    "Referer": "https://www.google.com/",
    "Sec-Ch-Ua": '"Google Chrome";v="147", "Not.A/Brand";v="8", "Chromium";v="147"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "cross-site",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
    ),
}

os.makedirs(output_directory, exist_ok=True)

if os.path.exists(requested_ids_file):
    with open(requested_ids_file, 'r') as file:
        requested_pano_ids = set(json.load(file))
else:
    requested_pano_ids = set()

# `requested_ids_file` is only flushed at end-of-pass / limit-hit, so a
# kill mid-pass loses the in-memory set. The CSV log is appended per-pano,
# and any successfully-saved JPEG names the pano too — both make
# crash-safe sources of truth. Union them in.
if os.path.exists(log_file):
    with open(log_file, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if row:
                requested_pano_ids.add(row[0])

if os.path.isdir(output_directory):
    for fname in os.listdir(output_directory):
        if fname.endswith('.jpg'):
            requested_pano_ids.add(fname[:-4])

print(f"resume: {len(requested_pano_ids)} panos already done")

if not os.path.exists(log_file):
    with open(log_file, 'w', newline='') as csvfile:
        log_writer = csv.writer(csvfile)
        log_writer.writerow(['pano_id', 'lat', 'lng'])


def crop_black_bars(img: Image.Image) -> Image.Image:
    """Trim solid-black borders left over from missing/failed edge tiles.
    Vectorized via numpy — handles partial/internal black tiles too, not just
    the right/bottom-edge case the simple max_x/max_y crop catches."""
    arr = np.asarray(img.convert("RGB"))
    nonblack = arr.sum(axis=2) > 0
    if not nonblack.any():
        return img
    rows = nonblack.any(axis=1)
    cols = nonblack.any(axis=0)
    rmin = int(rows.argmax())
    rmax = int(len(rows) - 1 - rows[::-1].argmax())
    cmin = int(cols.argmax())
    cmax = int(len(cols) - 1 - cols[::-1].argmax())
    if rmin == 0 and cmin == 0 and rmax == img.height - 1 and cmax == img.width - 1:
        return img
    return img.crop((cmin, rmin, cmax + 1, rmax + 1))


async def download_tile(session, url):
    async with session.get(url) as response:
        if response.status == 200:
            return await response.read()
        else:
            print(f"Failed to download tile from {url}")
            return None


async def download_and_stitch_tiles(pano_id, lat, lng, zoom=3, tile_size=512, i=0):
    """Download tiles, stitch into a panorama, crop black bars, save.
    Per-pano session: shared-session experiment dropped sustained throughput
    ~30% (stale keep-alives accumulated), so we open + close per pano."""
    start = datetime.datetime.now()
    num_horizontal_tiles = 1 << zoom
    num_vertical_tiles = num_horizontal_tiles // 2

    panorama_image = Image.new('RGB', (num_horizontal_tiles * tile_size, num_vertical_tiles * tile_size))
    got_any_tile = False

    async with aiohttp.ClientSession(headers=TILE_HEADERS) as session:
        # Schedule all 32 tile fetches concurrently — without create_task they
        # would be awaited one-at-a-time in the loop below, since a bare
        # coroutine doesn't start running until you await it.
        tasks = [
            (x, y, asyncio.create_task(
                download_tile(session, tile_url.format(pano_id=pano_id, zoom=zoom, x=x, y=y))
            ))
            for x in range(num_horizontal_tiles)
            for y in range(num_vertical_tiles)
        ]

        for x, y, task in tasks:
            tile_data = await task
            if tile_data:
                tile = Image.open(BytesIO(tile_data))
                panorama_image.paste(tile, (x * tile_size, y * tile_size))
                got_any_tile = True

    if not got_any_tile:
        print(f"Failed to create panorama for {pano_id}, no valid tiles found.")
        return

    cropped = crop_black_bars(panorama_image)
    cropped.save(os.path.join(output_directory, f'{pano_id}.jpg'))
    end = datetime.datetime.now()
    print(f"{i} : {pano_id} → {cropped.size[0]}x{cropped.size[1]} ({str(end-start).split(':')[-1]}s)")


async def main():
    request_count = 0

    for json_file in glob.glob(os.path.join(json_directory, '*.json')):
        # If the tracker file ever ends up under json_directory, don't try to
        # parse it as an input feed.
        if os.path.realpath(json_file) == os.path.realpath(requested_ids_file):
            continue
        with open(json_file, 'r') as file:
            data = json.load(file)
            for coordinate in data.get('customCoordinates', []):
                pano_id = coordinate['panoId']
                lat = coordinate['lat']
                lng = coordinate['lng']
                if pano_id not in requested_pano_ids and not str(pano_id).startswith('CAo'):
                    if request_count < request_limit:
                        request_count += 1
                        await download_and_stitch_tiles(pano_id, lat, lng, i=request_count)
                        requested_pano_ids.add(pano_id)
                        with open(log_file, 'a', newline='') as csvfile:
                            log_writer = csv.writer(csvfile)
                            log_writer.writerow([pano_id, lat, lng])
                    else:
                        with open(requested_ids_file, 'w') as file:
                            json.dump(list(requested_pano_ids), file)
                        print(f"Reached request limit of {request_limit}.")
                        return

    with open(requested_ids_file, 'w') as file:
        json.dump(list(requested_pano_ids), file)

    print("Completed downloading panoramas.")


if __name__ == "__main__":
    while True:
        try:
            asyncio.run(main())
        except:
            print("Waiting 10 seconds [error]")
            time.sleep(10)
