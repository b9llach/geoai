// ==UserScript==
// @name         AIMY Extension
// @namespace    http://tampermonkey.net/
// @version      3.32
// @description  PanoID → backend predict; per-pano dedupe + retry; smart map picker; shows current model epoch fetched from /api/v1/info.
// @author       billy
// @match        https://www.geoguessr.com/*
// @grant        GM_xmlhttpRequest
// @grant        GM.xmlHttpRequest
// @grant        unsafeWindow
// @run-at       document-start
// @require      https://unpkg.com/leaflet@1.9.4/dist/leaflet.js
// @connect      streetviewpixels-pa.googleapis.com
// @connect      cbk0.google.com
// @connect      192.168.0.12
// @connect      127.0.0.1
// @connect      localhost
// ==/UserScript==

(function() {
    'use strict';

    // Your geoai-serve instance (host:port). Everything else is derived from it.
    const SERVER = 'http://192.168.0.12:6301';
    const PREDICT_URL = SERVER + '/api/v1/predict';
    const PREDICT_STREAM_URL = SERVER + '/api/v1/predict_stream';
    const EXPLAIN_URL = SERVER + '/api/v1/explain';
    // Runtime toggleable via the in-overlay switch. Persisted to
     // localStorage. Initial defaults if nothing saved: auto-submit ON,
    // overlay expanded.
    let _autoSubmit = (() => {
        try { const v = localStorage.getItem('aimy-autoguess'); return v === null ? true : v === '1'; }
        catch (e) { return true; }
    })();
    let _collapsed = (() => {
        try { return localStorage.getItem('aimy-collapsed') === '1'; }
        catch (e) { return false; }
    })();
    // Prediction mode, sent as `cascade` on every predict POST. Two modes:
    // 'fast' (Stage 1 = ProtoNet-select) and 'refined' (+ Stage 2 OCR/VLM).
    let _cascade = (() => {
        // Only two modes now: 'fast' (Stage 1 = ProtoNet-select) and 'refined'
        // (+ Stage 2). Migrate any legacy value (plain/country_only/joint/fancy)
        // to 'fast' — they all collapse to the same Stage-1 path server-side.
        try {
            const v = localStorage.getItem('aimy-cascade');
            return v === 'refined' ? 'refined' : 'fast';
        } catch (e) { return 'fast'; }
    })();
    const DEBUG_NET   = false; // true → log every Geoguessr-internal fetch/XHR with body+response
    const ZOOM = 3;          // 8x4 = 32 tiles, ~4096x2048 stitched
    const TILE_SIZE = 512;

    // unsafeWindow gives us the page's REAL window — bypassing Tampermonkey's
    // sandbox proxy. Needed to access google.maps and patch the actual fetch
    // the page uses (Geoguessr's CSP blocks inline <script> injection).
    const W = (typeof unsafeWindow !== 'undefined') ? unsafeWindow : window;


    // Patch google.maps.Map in PAGE context to capture each Map instance.
    // Polling lost the race on slow loads. Instead: hook the moment google
    // gets assigned to window, then the moment .maps gets assigned to that,
    // then wrap .Map. This catches the API load before any Map is created.
    function wrapMapClass(MapCls) {
        if (!MapCls || MapCls.__aimy_patched) return MapCls;
        function Wrapped(...args) {
            const map = new MapCls(...args);
            (W.__aimy_maps = W.__aimy_maps || []).push(map);
            return map;
        }
        Wrapped.prototype = MapCls.prototype;
        Object.assign(Wrapped, MapCls);
        Wrapped.__aimy_patched = true;
        return Wrapped;
    }

    (function setupMapsHook() {
        // Attempt 1: if Maps already loaded, patch in place.
        try {
            if (W.google && W.google.maps && W.google.maps.Map) {
                W.google.maps.Map = wrapMapClass(W.google.maps.Map);
                console.log('[aimy] Map class patched in place');
            }
        } catch (e) {}

        // Attempt 2: property setter on google → maps → Map.
        try {
            if (!Object.getOwnPropertyDescriptor(W, 'google') ||
                Object.getOwnPropertyDescriptor(W, 'google').configurable) {
                let _google = W.google;
                Object.defineProperty(W, 'google', {
                    configurable: true,
                    get() { return _google; },
                    set(v) {
                        _google = v;
                        if (!v || v.__aimy_hooked) return;
                        v.__aimy_hooked = true;
                        let _maps = v.maps;
                        if (_maps && _maps.Map && !_maps.Map.__aimy_patched) {
                            _maps.Map = wrapMapClass(_maps.Map);
                        }
                        try {
                            Object.defineProperty(v, 'maps', {
                                configurable: true,
                                get() { return _maps; },
                                set(m) {
                                    _maps = m;
                                    if (m && m.Map && !m.Map.__aimy_patched) {
                                        let _MapClass = wrapMapClass(m.Map);
                                        try {
                                            Object.defineProperty(m, 'Map', {
                                                configurable: true,
                                                get() { return _MapClass; },
                                                set(c) { _MapClass = wrapMapClass(c); }
                                            });
                                        } catch (e2) {}
                                        console.log('[aimy] Map patched via property hook');
                                    }
                                }
                            });
                        } catch (e2) {}
                    }
                });
            }
        } catch (e) { /* property already non-configurable */ }

        // Attempt 3: keep polling forever in case 1 & 2 missed. Cheap.
        // Also patches the Map.prototype methods so any USE of an existing
        // Map (created before our script ran, or via a constructor-bypass)
        // captures the instance into __aimy_maps. This is the strongest of
        // the four because it doesn't depend on catching the constructor.
        const pollInt = setInterval(() => {
            try {
                if (W.google && W.google.maps && W.google.maps.Map &&
                    !W.google.maps.Map.__aimy_patched) {
                    W.google.maps.Map = wrapMapClass(W.google.maps.Map);
                    console.log('[aimy] Map class patched via poll');
                }
                hookMapPrototype();
                hookStreetViewPanoramaPrototype();
            } catch (e) {}
        }, 250);
        setTimeout(() => clearInterval(pollInt), 60_000);
    })();

    // Patch instance methods so the instance gets tracked the moment it's
    // used — independent of how it was constructed. Geoguessr's guess map
    // calls setCenter / panTo / addListener routinely, so this catches it
    // even if every constructor-level hook missed.
    function hookMapPrototype() {
        if (!W.google || !W.google.maps || !W.google.maps.Map) return;
        const proto = W.google.maps.Map.prototype;
        if (proto.__aimy_proto_hooked) return;
        proto.__aimy_proto_hooked = true;
        // Methods that prove a Map instance is in active use (track-or-skip).
        const methods = ['setCenter', 'panTo', 'setZoom', 'fitBounds'];
        for (const m of methods) {
            const orig = proto[m];
            if (typeof orig !== 'function') continue;
            proto[m] = function(...args) {
                if (!this.__aimy_tracked) {
                    this.__aimy_tracked = true;
                    (W.__aimy_maps = W.__aimy_maps || []).push(this);
                    console.log(`[aimy] Map tracked via .${m}() prototype hook`);
                }
                return orig.apply(this, args);
            };
        }
        // addListener is special — we ALSO note when 'click' is registered.
        // The interactive guess map registers 'click' fresh each round; the
        // read-only results-display map does not. Picking the map with the
        // most recent 'click' listener ensures we trigger placePin on the
        // right instance.
        const origAddListener = proto.addListener;
        if (typeof origAddListener === 'function') {
            proto.addListener = function(eventName) {
                if (!this.__aimy_tracked) {
                    this.__aimy_tracked = true;
                    (W.__aimy_maps = W.__aimy_maps || []).push(this);
                    console.log('[aimy] Map tracked via .addListener() prototype hook');
                }
                if (eventName === 'click') {
                    this.__aimy_click_listener_at = Date.now();
                }
                return origAddListener.apply(this, arguments);
            };
        }
    }

    // Hook StreetViewPanorama.prototype.setPano so we catch pano-change events
    // when Geoguessr re-uses the same widget across duels rounds (the metadata
    // RPC only fires on the first pano; subsequent rounds just call setPano on
    // the existing panorama instance).
    function hookStreetViewPanoramaPrototype() {
        if (!W.google || !W.google.maps || !W.google.maps.StreetViewPanorama) return;
        const proto = W.google.maps.StreetViewPanorama.prototype;
        if (proto.__aimy_sv_hooked) return;
        proto.__aimy_sv_hooked = true;
        const origSetPano = proto.setPano;
        if (typeof origSetPano === 'function') {
            proto.setPano = function(panoID) {
                // Remember the most-recent panorama instance so we can later
                // query its `getPano()` to find the *currently visible* pano
                // (used to identify the round-start pano after the gate
                // clears, since by then no fresh setPano typically fires).
                _activePanorama = this;
                if (typeof panoID === 'string' && panoID.length > 5) {
                    onPanoIDDetected(panoID).catch(e =>
                        console.warn('[aimy] onPanoIDDetected (setPano) threw:', e));
                }
                return origSetPano.apply(this, arguments);
            };
            console.log('[aimy] StreetViewPanorama.setPano hooked');
        }
    }

    // Most-recent StreetViewPanorama instance, captured by the setPano
    // prototype hook. Used to ask "what pano is the user currently looking
    // at?" — the canonical signal for which pano to predict on, regardless
    // of whether a fresh setPano event has fired recently.
    let _activePanorama = null;
    function getCurrentPanoFromPanorama() {
        if (!_activePanorama) return null;
        try {
            const p = _activePanorama.getPano && _activePanorama.getPano();
            return (typeof p === 'string' && p.length > 5) ? p : null;
        } catch (e) { return null; }
    }

    // Last-resort fallback: walk the DOM for existing Map instances stored on
    // `.gm-style` parent divs as `__gm` (Google's internal property). Used when
    // none of the constructor patches caught the Map.
    function findMapInDom() {
        const stylis = document.querySelectorAll('.gm-style');
        const isMap = (v) => v && typeof v === 'object' &&
            typeof v.panTo === 'function' &&
            typeof v.setCenter === 'function' &&
            typeof v.getCenter === 'function';
        for (const el of stylis) {
            let node = el;
            for (let d = 0; d < 6 && node; d++) {
                for (const k of Object.getOwnPropertyNames(node)) {
                    const v = node[k];
                    if (isMap(v)) return v;
                    if (v && typeof v === 'object' && isMap(v.map)) return v.map;
                    if (v && typeof v === 'object' && isMap(v.gm_map)) return v.gm_map;
                }
                node = node.parentElement;
            }
        }
        return null;
    }

    // ── Network sniffer (DEBUG_NET) ────────────────────────────────────
    // Patch via unsafeWindow so we hook the page's REAL fetch/XHR (CSP
    // blocks inline-script injection, so we can't run code in page context
    // any other way). Logs Geoguessr-internal calls only.
    if (DEBUG_NET) {
        const isInteresting = (u) =>
            /geoguessr\.com\/api/.test(u) ||
            /game-server\.geoguessr\.com/.test(u);

        const origFetch = W.fetch.bind(W);
        W.fetch = async function(input, init) {
            const url = typeof input === 'string' ? input : (input && input.url) || '';
            const method = (init && init.method) || (input && input.method) || 'GET';
            if (isInteresting(url)) {
                let body = init && init.body;
                if (body && typeof body !== 'string') body = '[non-string body]';
                console.log('%c[net] ' + method + ' ' + url, 'color:#9cf',
                    body ? 'body=' + String(body).slice(0, 400) : '');
            }
            const resp = await origFetch.apply(W, arguments);
            if (isInteresting(url)) {
                resp.clone().text()
                    .then(t => console.log('%c[net]   <- ' + resp.status + ' ' + url, 'color:#9c9', t.slice(0, 600)))
                    .catch(() => {});
            }
            return resp;
        };

        const origOpen = W.XMLHttpRequest.prototype.open;
        const origSend = W.XMLHttpRequest.prototype.send;
        W.XMLHttpRequest.prototype.open = function(method, url) {
            this.__aimy_m = method;
            this.__aimy_u = url;
            return origOpen.apply(this, arguments);
        };
        W.XMLHttpRequest.prototype.send = function(body) {
            const url = this.__aimy_u || '';
            if (isInteresting(url)) {
                console.log('%c[net-xhr] ' + this.__aimy_m + ' ' + url, 'color:#fc9',
                    body ? 'body=' + String(body).slice(0, 400) : '');
                this.addEventListener('load', () => {
                    console.log('%c[net-xhr]   <- ' + this.status + ' ' + url, 'color:#9c9',
                        (this.responseText || '').slice(0, 600));
                });
            }
            return origSend.apply(this, arguments);
        };
        console.log('[aimy] DEBUG_NET hooks installed via unsafeWindow');
    }

    let globalPanoID = undefined;
    let roundNumber = 1;

    // Dedupe predict + autoGuess by panoID. Geoguessr re-fires the metadata
    // RPC for the same pano during the results screen of duels (and similar
    // mid-round scenarios), which without dedupe spawns repeated predict +
    // autoGuess attempts. Bounded set; keeps last 50 panoIDs to avoid leaks.
    const _predictedPanoIDs = new Set();
    const MAX_TRACKED_PANOS = 100;
    let _currentAutoGuessPano = null;  // tracks the active retry's pano for abort-on-new-round

    // Cross-mode round-detection: ALL the gate/state-machine logic was ripped
    // out (it kept breaking across game modes). Replaced with a simpler
    // per-pano flow inside onPanoIDDetected:
    //   1. Wait for the guess button to appear (= "live round").
    //   2. Verify the panorama is STILL on this pano (via getPano()) — if
    //      it's moved on, this pano-id event is stale, abandon.
    //   3. Predict + autoGuess.
    // Each pano-id event runs independently. Intermediate walk panos
    // naturally fall out because by the time the button reappears, the
    // panorama has moved past them.
    const PANO_BUTTON_WAIT_MS = 90_000;  // max wait for button per pano

    function isNewGame() { return roundNumber === 1; }
    function getGameID() { return window.location.pathname.split('/')[2]; }
    async function wait(ms) { return new Promise(r => setTimeout(r, ms)); }

    // Team-duels uses gs2.geoguessr.com/{sessionId-32hex}/{roundId-24hex}/guess
    // and the page only knows those IDs from a separate state-fetch call.
    // Hook fetch/XHR via unsafeWindow to capture the latest pair so submitGuess
    // can construct the URL when it's time to guess.
    const _teamDuelsCtx = { sessionId: null, roundId: null };
    (function hookTeamDuelsIds() {
        const re = /gs2\.geoguessr\.com\/([a-f0-9]{32})\/([a-f0-9]{24})\b/;
        const stash = (url) => {
            const m = re.exec(url || '');
            if (m) {
                _teamDuelsCtx.sessionId = m[1];
                _teamDuelsCtx.roundId = m[2];
            }
        };
        try {
            const origFetch = W.fetch.bind(W);
            W.fetch = function(input, init) {
                stash(typeof input === 'string' ? input : (input && input.url));
                return origFetch.apply(W, arguments);
            };
            const origOpen = W.XMLHttpRequest.prototype.open;
            W.XMLHttpRequest.prototype.open = function(method, url) {
                stash(url);
                return origOpen.apply(this, arguments);
            };
        } catch (e) {
            console.warn('[aimy] failed to hook team-duels ID sniffer:', e);
        }
    })();

    let _map = null, _finalMarker = null;
    // Inline map is hidden by default and created lazily on first reveal.
    let _mapOn = (() => {
        try { return localStorage.getItem('aimy-map-on') === '1'; }
        catch (e) { return false; }
    })();
    let _lastLat = null, _lastLng = null;

    const PIN_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="34" viewBox="0 0 24 34">' +
        '<path d="M12 0C5.4 0 0 5.4 0 12c0 9 12 22 12 22s12-13 12-22c0-6.6-5.4-12-12-12z" ' +
        'fill="#ef4444" stroke="#fff" stroke-width="1.6"/>' +
        '<circle cx="12" cy="12" r="4" fill="#fff"/>' +
        '</svg>';

    function ensureLeafletCSS() {
        if (document.getElementById('aimy-leaflet-css')) return;
        const link = document.createElement('link');
        link.id = 'aimy-leaflet-css';
        link.rel = 'stylesheet';
        link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
        document.head.appendChild(link);
    }

    function injectStyles() {
        if (document.getElementById('aimy-styles')) return;
        const s = document.createElement('style');
        s.id = 'aimy-styles';
        s.textContent = `
            #aimy-overlay {
                position: fixed; top: 14px; right: 14px; z-index: 999999;
                width: 268px;
                background: rgba(22,24,30,0.94);
                -webkit-backdrop-filter: blur(8px); backdrop-filter: blur(8px);
                color: #e8eaed; border-radius: 12px;
                box-shadow: 0 10px 34px rgba(0,0,0,0.5),
                            0 0 0 1px rgba(255,255,255,0.06);
                font-family: 'Neue Helvetica','Helvetica Neue',-apple-system,
                    BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
                font-size: 12px; line-height: 1.4;
                overflow: hidden; user-select: none;
                -webkit-font-smoothing: antialiased;
            }
            #aimy-header {
                display: flex; align-items: center; justify-content: space-between;
                height: 34px; padding: 0 7px 0 11px; cursor: grab;
                border-bottom: 1px solid rgba(255,255,255,0.06);
            }
            #aimy-header:active { cursor: grabbing; }
            /* Empty flexible drag region on the header's left side. */
            #aimy-grip { flex: 1; align-self: stretch; }
            .aimy-hcontrols { display: flex; align-items: center; gap: 8px; }
            #aimy-cascade {
                background: rgba(255,255,255,0.07); color: #d6d9df;
                border: 1px solid rgba(255,255,255,0.12); border-radius: 6px;
                font: 600 10.5px/1 inherit; padding: 3px 5px;
                outline: none; cursor: pointer;
            }
            #aimy-cascade option { background: #1c1e24; color: #e8eaed; }
            .aimy-switch {
                display: inline-flex; align-items: center; gap: 6px; cursor: pointer;
                font-size: 10.5px; font-weight: 600;
                color: rgba(232,234,237,0.55); transition: color 140ms;
            }
            .aimy-switch:hover { color: rgba(232,234,237,0.85); }
            .aimy-switch .aimy-dot {
                width: 22px; height: 13px; border-radius: 999px;
                background: rgba(255,255,255,0.12); position: relative;
                transition: background 180ms cubic-bezier(.16,1,.3,1);
            }
            .aimy-switch .aimy-dot::after {
                content: ""; position: absolute; top: 2px; left: 2px;
                width: 9px; height: 9px; border-radius: 50%; background: #cfd2d8;
                transition: transform 180ms cubic-bezier(.16,1,.3,1), background 180ms;
            }
            .aimy-switch.on { color: #e8eaed; }
            .aimy-switch.on .aimy-dot { background: #6cbe3f; }
            .aimy-switch.on .aimy-dot::after { transform: translateX(9px); background: #fff; }
            #aimy-collapse {
                width: 22px; height: 22px; display: flex; align-items: center;
                justify-content: center; color: rgba(232,234,237,0.5);
                cursor: pointer; border-radius: 6px;
                transition: background 120ms, color 120ms,
                            transform 220ms cubic-bezier(.16,1,.3,1);
            }
            #aimy-collapse:hover { background: rgba(255,255,255,0.06); color: #e8eaed; }
            #aimy-collapse svg { width: 12px; height: 12px; }

            #aimy-body { padding: 11px 13px 12px; user-select: text; }
            .aimy-place {
                font-size: 13px; font-weight: 600; color: #f1f3f6; line-height: 1.35;
            }
            .aimy-coords {
                font-family: ui-monospace,'SF Mono',Menlo,monospace;
                font-size: 11px; color: #7f8794; margin-top: 4px;
                font-feature-settings: "tnum"; letter-spacing: -0.01em;
            }
            .aimy-fallback {
                display: inline-block; margin-left: 6px; padding: 1px 6px;
                background: rgba(245,158,11,0.14); color: #f6b73c;
                border-radius: 5px; font-size: 9.5px; font-weight: 600;
                vertical-align: middle;
            }
            .aimy-s2-badge {
                display: inline-block; margin-left: 6px; padding: 1px 6px;
                background: rgba(108,190,63,0.16); color: #8fd25e;
                border-radius: 5px; font-size: 9.5px; font-weight: 700;
                vertical-align: middle; text-transform: uppercase; letter-spacing: 0.04em;
            }
            .aimy-s2-status {
                margin-top: 8px; padding: 5px 8px;
                background: rgba(108,190,63,0.09); border-radius: 6px;
                font-size: 10.5px; color: #8fd25e;
                font-family: ui-monospace,'SF Mono',Menlo,monospace;
                animation: aimy-s2-pulse 1.4s ease-in-out infinite;
            }
            @keyframes aimy-s2-pulse { 0%,100% { opacity: .85; } 50% { opacity: .45; } }
            .aimy-s2-explain {
                margin-top: 8px; padding: 6px 8px;
                background: rgba(108,190,63,0.06);
                border-left: 2px solid rgba(108,190,63,0.4); border-radius: 4px;
                font-size: 10.5px; color: #9aa3b3; line-height: 1.4; font-style: italic;
            }
            .aimy-btnrow { display: flex; gap: 7px; margin-top: 11px; }
            #aimy-maptoggle, #aimy-explain {
                appearance: none; flex: 1;
                background: rgba(255,255,255,0.05); color: #aeb4c0;
                border: 1px solid rgba(255,255,255,0.08); border-radius: 7px;
                font: 600 10.5px/1 inherit; padding: 7px; cursor: pointer;
                letter-spacing: 0.04em; text-transform: uppercase;
                transition: background 120ms, color 120ms;
            }
            #aimy-maptoggle:hover, #aimy-explain:hover {
                background: rgba(255,255,255,0.09); color: #e8eaed;
            }
            #aimy-explain { color: #8fd25e; border-color: rgba(108,190,63,0.25); }
            #aimy-explain:hover { background: rgba(108,190,63,0.12); color: #a7e072; }
            #aimy-explain.busy { opacity: 0.6; cursor: wait; }

            /* Fullscreen heatmap viewer */
            #aimy-lightbox {
                position: fixed; inset: 0; z-index: 2147483000;
                display: none; align-items: center; justify-content: center;
                flex-direction: column; gap: 14px;
                background: rgba(6,7,10,0.86); -webkit-backdrop-filter: blur(3px);
                backdrop-filter: blur(3px); cursor: zoom-out;
            }
            #aimy-lightbox.on { display: flex; }
            #aimy-lightbox img {
                max-width: 96vw; max-height: 78vh; border-radius: 8px;
                box-shadow: 0 12px 50px rgba(0,0,0,0.7);
                image-rendering: auto;
            }
            #aimy-lightbox .aimy-lb-cap {
                color: #cfd3da; font: 600 12px/1.4 'Helvetica Neue',system-ui,sans-serif;
                letter-spacing: 0.02em; text-align: center; max-width: 90vw;
            }
            #aimy-lightbox .aimy-lb-cap b { color: #8fd25e; }
            #aimy-lightbox .aimy-lb-spin {
                color: #8fd25e; font: 600 13px/1 system-ui,sans-serif;
                animation: aimy-s2-pulse 1.2s ease-in-out infinite;
            }
            #aimy-map {
                margin-top: 9px; height: 168px; border-radius: 8px;
                overflow: hidden; background: #0d0f14; display: none;
            }
            #aimy-overlay.map-on #aimy-map { display: block; }

            /* ─── Collapsed state: shrink to a small round launcher ─────── */
            #aimy-fab {
                display: none;
                width: 44px; height: 44px; border-radius: 50%;
                align-items: center; justify-content: center;
                cursor: pointer; color: #6cbe3f;
                background: rgba(22,24,30,0.96);
                box-shadow: 0 6px 20px rgba(0,0,0,0.5),
                            0 0 0 1px rgba(108,190,63,0.45) inset;
                transition: transform 140ms cubic-bezier(.16,1,.3,1), color 140ms;
            }
            #aimy-fab:hover { transform: scale(1.06); color: #8fd25e; }
            #aimy-fab:active { cursor: grabbing; }
            #aimy-fab svg { width: 20px; height: 20px; }

            /* When collapsed, the card chrome vanishes and only the circle
               remains — the overlay itself becomes transparent and auto-sized
               so it's just the 44px launcher. */
            #aimy-overlay.aimy-collapsed {
                width: auto; background: transparent; overflow: visible;
                box-shadow: none; -webkit-backdrop-filter: none; backdrop-filter: none;
            }
            #aimy-overlay.aimy-collapsed #aimy-header,
            #aimy-overlay.aimy-collapsed #aimy-body { display: none; }
            #aimy-overlay.aimy-collapsed #aimy-fab { display: flex; }
        `;
        document.head.appendChild(s);
    }

    function makeDraggable(handles, target) {
        // One-time cleanup of a stale key from an older resizable build —
        // a leftover inline width/height would pin the overlay's size and
        // stop it shrinking to the collapsed circle.
        try { localStorage.removeItem('aimy-overlay-size'); } catch (e) {}

        // Restore last position from localStorage (per-domain).
        try {
            const saved = JSON.parse(localStorage.getItem('aimy-overlay-pos') || 'null');
            if (saved && typeof saved.x === 'number' && typeof saved.y === 'number') {
                target.style.left = `${saved.x}px`;
                target.style.top  = `${saved.y}px`;
                target.style.right = 'auto';
            }
        } catch (e) {}

        let dx = 0, dy = 0, dragging = false, moved = false, activeHandle = null;
        for (const handle of [].concat(handles)) {
            handle.addEventListener('mousedown', (e) => {
                if (e.target.closest('#aimy-collapse')) return;
                if (e.target.closest('.aimy-switch')) return;
                dragging = true; moved = false; activeHandle = handle;
                const r = target.getBoundingClientRect();
                dx = e.clientX - r.left;
                dy = e.clientY - r.top;
                e.preventDefault();
            });
        }
        document.addEventListener('mousemove', (e) => {
            if (!dragging) return;
            moved = true;
            const x = Math.max(0, Math.min(window.innerWidth  - target.offsetWidth,  e.clientX - dx));
            const y = Math.max(0, Math.min(window.innerHeight - target.offsetHeight, e.clientY - dy));
            target.style.left = `${x}px`;
            target.style.top  = `${y}px`;
            target.style.right = 'auto';
        });
        document.addEventListener('mouseup', () => {
            if (!dragging) return;
            dragging = false;
            // Flag a real drag on the launcher so its click handler doesn't
            // also expand the card after a reposition.
            if (moved && activeHandle && activeHandle.id === 'aimy-fab') {
                activeHandle.dataset.dragged = '1';
            }
            activeHandle = null;
            if (!moved) return;
            try {
                const r = target.getBoundingClientRect();
                localStorage.setItem('aimy-overlay-pos', JSON.stringify({ x: r.left, y: r.top }));
            } catch (e) {}
        });
    }

    function ensureOverlay() {
        let el = document.getElementById('aimy-overlay');
        if (el) return el;
        ensureLeafletCSS();
        injectStyles();
        el = document.createElement('div');
        el.id = 'aimy-overlay';
        el.innerHTML = `
            <div id="aimy-header">
                <span id="aimy-grip" title="drag"></span>
                <div class="aimy-hcontrols">
                    <select id="aimy-cascade" title="prediction mode">
                        <option value="fast">Stage 1</option>
                        <option value="refined">+ Stage 2</option>
                    </select>
                    <span class="aimy-switch" id="aimy-auto" title="toggle auto-guess">
                        <span>Auto</span><span class="aimy-dot"></span>
                    </span>
                    <span id="aimy-collapse" title="collapse / expand">
                        <svg viewBox="0 0 14 14" fill="none">
                            <path d="M3 5l4 4 4-4" stroke="currentColor" stroke-width="1.8"
                                  stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                    </span>
                </div>
            </div>
            <div id="aimy-body">
                <div class="aimy-place" id="aimy-place">—</div>
                <div class="aimy-coords" id="aimy-coords"></div>
                <div class="aimy-s2-status" id="aimy-s2-status" style="display:none;"></div>
                <div class="aimy-s2-explain" id="aimy-s2-explain" style="display:none;"></div>
                <div class="aimy-btnrow">
                    <button id="aimy-maptoggle">show map</button>
                    <button id="aimy-explain" title="show what the model looked at">explain</button>
                </div>
                <div id="aimy-map"></div>
            </div>
            <div id="aimy-fab" title="open AIMY">
                <svg viewBox="0 0 24 24" fill="none">
                    <path d="M12 2C8.1 2 5 5.1 5 9c0 5.2 7 13 7 13s7-7.8 7-13c0-3.9-3.1-7-7-7z"
                          fill="currentColor" stroke="rgba(0,0,0,0.35)" stroke-width="1"/>
                    <circle cx="12" cy="9" r="2.6" fill="#15171e"/>
                </svg>
            </div>
        `;
        document.body.appendChild(el);

        if (_collapsed) el.classList.add('aimy-collapsed');
        if (_mapOn) el.classList.add('map-on');

        const setCollapsed = (v) => {
            _collapsed = v;
            el.classList.toggle('aimy-collapsed', _collapsed);
            try { localStorage.setItem('aimy-collapsed', _collapsed ? '1' : '0'); } catch (e2) {}
            if (!_collapsed && _mapOn && _map) setTimeout(() => _map.invalidateSize(), 60);
        };

        const collapseBtn = document.getElementById('aimy-collapse');
        collapseBtn.onclick = (e) => { e.stopPropagation(); setCollapsed(true); };

        // The collapsed launcher: a click re-opens the card. A drag (handled
        // by makeDraggable below) repositions it without triggering expand.
        const fab = document.getElementById('aimy-fab');
        fab.onclick = (e) => {
            e.stopPropagation();
            if (fab.dataset.dragged === '1') { fab.dataset.dragged = ''; return; }
            setCollapsed(false);
        };

        const autoBtn = document.getElementById('aimy-auto');
        const syncAutoBtn = () => autoBtn.classList.toggle('on', _autoSubmit);
        syncAutoBtn();
        autoBtn.onclick = (e) => {
            e.stopPropagation();
            _autoSubmit = !_autoSubmit;
            try { localStorage.setItem('aimy-autoguess', _autoSubmit ? '1' : '0'); } catch (e2) {}
            syncAutoBtn();
            console.log(`[aimy] autoguess ${_autoSubmit ? 'ENABLED' : 'DISABLED'}`);
        };

        const cascadeSel = document.getElementById('aimy-cascade');
        if (cascadeSel) {
            cascadeSel.value = _cascade;
            cascadeSel.onmousedown = (e) => e.stopPropagation();
            cascadeSel.onclick = (e) => e.stopPropagation();
            cascadeSel.onchange = (e) => {
                e.stopPropagation();
                _cascade = cascadeSel.value;
                try { localStorage.setItem('aimy-cascade', _cascade); } catch (e2) {}
                console.log(`[aimy] mode → ${_cascade}`);
            };
        }

        // Inline map toggle — created lazily on first show (Leaflet needs a
        // sized, visible container), centered on the latest prediction.
        const mapToggle = document.getElementById('aimy-maptoggle');
        const syncMapToggle = () => { mapToggle.textContent = _mapOn ? 'hide map' : 'show map'; };
        syncMapToggle();
        mapToggle.onclick = (e) => {
            e.stopPropagation();
            _mapOn = !_mapOn;
            el.classList.toggle('map-on', _mapOn);
            try { localStorage.setItem('aimy-map-on', _mapOn ? '1' : '0'); } catch (e2) {}
            syncMapToggle();
            if (_mapOn) ensureMap();
        };

        // Explain: ask the server for the occlusion heatmap of the current
        // pano and show it in a fullscreen viewer.
        const explainBtn = document.getElementById('aimy-explain');
        explainBtn.onclick = async (e) => {
            e.stopPropagation();
            const pid = globalPanoID;
            if (!pid) { openLightbox(null, 'no pano detected yet — load a round first'); return; }
            if (explainBtn.classList.contains('busy')) return;
            explainBtn.classList.add('busy');
            openLightbox(null, 'reading the pano…');
            try {
                const { blob, headers } = await gmPostBlob(EXPLAIN_URL, { panoID: pid });
                const sim = (headers.match(/x-explain-sim:\s*([\d.]+)/i) || [])[1];
                const url = URL.createObjectURL(blob);
                openLightbox(url, sim
                    ? `prototype match <b>${sim}</b> — brighter red = the model relied on it more`
                    : 'brighter red = the model relied on that region more');
            } catch (err) {
                openLightbox(null, 'explain failed: ' + (err.message || err));
            } finally {
                explainBtn.classList.remove('busy');
            }
        };

        makeDraggable([document.getElementById('aimy-header'), fab], el);
        return el;
    }

    // Lazily build the inline Leaflet map (only when revealed — Leaflet needs
    // a sized, visible container) and point it at the latest prediction.
    function ensureMap() {
        if (typeof L === 'undefined') return;
        if (_lastLat == null || _lastLng == null) return;
        if (!_map) {
            _map = L.map('aimy-map', {
                zoomControl: true, attributionControl: false,
            }).setView([_lastLat, _lastLng], 6);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                maxZoom: 19,
            }).addTo(_map);
        }
        if (_finalMarker) _finalMarker.setLatLng([_lastLat, _lastLng]);
        else {
            const icon = L.divIcon({
                className: '', html: PIN_SVG,
                iconSize: [24, 34], iconAnchor: [12, 34],
            });
            _finalMarker = L.marker([_lastLat, _lastLng], { icon }).addTo(_map);
        }
        _map.setView([_lastLat, _lastLng], 6);
        setTimeout(() => _map && _map.invalidateSize(), 50);
    }

    // ─── Fullscreen heatmap viewer (the /explain overlay) ────────────────
    let _lbUrl = null;
    function ensureLightbox() {
        let lb = document.getElementById('aimy-lightbox');
        if (lb) return lb;
        lb = document.createElement('div');
        lb.id = 'aimy-lightbox';
        lb.innerHTML =
            '<div class="aimy-lb-spin">reading…</div>' +
            '<img alt="model attention" style="display:none;">' +
            '<div class="aimy-lb-cap"></div>';
        lb.onclick = () => closeLightbox();
        document.body.appendChild(lb);
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeLightbox();
        });
        return lb;
    }
    function openLightbox(url, captionHTML) {
        const lb = ensureLightbox();
        const img = lb.querySelector('img');
        const spin = lb.querySelector('.aimy-lb-spin');
        const cap = lb.querySelector('.aimy-lb-cap');
        if (_lbUrl) { URL.revokeObjectURL(_lbUrl); _lbUrl = null; }
        if (url) {
            _lbUrl = url;
            img.src = url; img.style.display = '';
            spin.style.display = 'none';
        } else {
            img.removeAttribute('src'); img.style.display = 'none';
            spin.style.display = '';
        }
        cap.innerHTML = captionHTML || '';
        lb.classList.add('on');
    }
    function closeLightbox() {
        const lb = document.getElementById('aimy-lightbox');
        if (lb) lb.classList.remove('on');
        if (_lbUrl) { URL.revokeObjectURL(_lbUrl); _lbUrl = null; }
    }

    function showPrediction(data, round, panoID) {
        ensureOverlay();
        const lat = data.lat ?? data.final_lat;
        const lng = data.lng ?? data.final_lng;
        if (typeof lat !== 'number' || typeof lng !== 'number') return;
        _lastLat = lat; _lastLng = lng;

        // Coords + reverse-geocoded name.
        document.getElementById('aimy-coords').textContent =
            `${lat.toFixed(5)}, ${lng.toFixed(5)}`;
        const placeBits = [data.admin2, data.admin1, data.country].filter(Boolean);
        const fb = data.fallback_used
            ? ` <span class="aimy-fallback">L9 fallback</span>` : '';
        // Stage 2 precision badge: only when refined cascade actually used the refinement.
        let s2Badge = '';
        if (data.stage2_used) {
            s2Badge = ` <span class="aimy-s2-badge" title="Stage 2 refined ` +
                `(${(data.stage2_precision || 'city')})">S2·${data.stage2_precision || 'city'}</span>`;
        }
        // ProtoNet-selector badge: shows it picked this L9 cell by image-feature
        // match (over the top-K candidates) and how confident the match was.
        let pnBadge = '';
        if (data.protonet_select && data.protonet_select.selected) {
            const sim = (data.protonet_select.top_sim ?? 0).toFixed(2);
            const k = data.protonet_select.select_k ?? '';
            pnBadge = ` <span class="aimy-s2-badge" title="ProtoNet selected this L9 cell ` +
                `by image similarity over top-${k} candidates">PN·${sim}</span>`;
        }
        document.getElementById('aimy-place').innerHTML =
            (placeBits.length ? placeBits.join(', ') : '—') + fb + s2Badge + pnBadge;

        // Stage 2 explanation panel: visible when Stage 2 actually
        // refined the prediction. When Stage 2 defers to Stage 1, that
        // usually means Stage 1's cell-head guess is already as specific
        // as the image evidence supports — so we hide the panel rather
        // than display a confusing "no refinement" message.
        const expEl = document.getElementById('aimy-s2-explain');
        if (expEl) {
            if (data.stage2_used && data.stage2_explanation) {
                expEl.textContent = data.stage2_explanation;
                expEl.style.display = '';
            } else if (data.stage2_error) {
                // Real error path: show it (helps debugging).
                expEl.textContent = 'Stage 2 error: ' + data.stage2_error;
                expEl.style.display = '';
            } else {
                expEl.style.display = 'none';
            }
        }
        // Update the inline map only when it's currently revealed.
        if (_mapOn) ensureMap();

        // writeText returns a Promise; sync try/catch misses the rejection
        // that fires when the document isn't focused. Swallow via .catch.
        try {
            const p = navigator.clipboard && navigator.clipboard.writeText(
                `${lat.toFixed(5)}, ${lng.toFixed(5)}`);
            if (p && typeof p.catch === 'function') p.catch(() => {});
        } catch (e) {}
    }

    // Drive the official UI flow: trigger a click on the guess Map at the
    // predicted lat/lng (Geoguessr's React listener catches it, drops the
    // pin, enables the Guess button), then click the Guess button. Same
    // code path as a manual guess, so React state updates on its own — no
    // reload needed.
    // Single funnel for "we just saw a fresh pano". Called from both the
    // XHR/metadata RPC hook AND StreetViewPanorama.setPano. Dedupe handles
    // duplicate fires from either source.
    // Polls the DOM for the live-guess button to appear. Returns when the
    // button is visible+enabled, OR after maxMs elapses (returning false).
    // The button is GG's authoritative signal that "you can submit a guess
    // right now" — i.e. we're in a live round, not the result screen or
    // an inter-round animation.
    async function waitForGuessButton(maxMs) {
        const t0 = Date.now();
        while (Date.now() - t0 < maxMs) {
            if (findGuessButton()) return true;
            await wait(300);
        }
        return false;
    }

    async function onPanoIDDetected(panoID) {
        if (typeof panoID !== 'string' || !panoID) return;
        if (_predictedPanoIDs.has(panoID)) return;
        _predictedPanoIDs.add(panoID);
        if (_predictedPanoIDs.size > MAX_TRACKED_PANOS) {
            const arr = Array.from(_predictedPanoIDs);
            _predictedPanoIDs.clear();
            arr.slice(-50).forEach(id => _predictedPanoIDs.add(id));
        }

        globalPanoID = panoID;
        console.log(`[aimy] pano detected: ${panoID.slice(0,8)}…`);

        // Wait for the guess button to appear. Multiple parallel calls (one
        // per pano during a duels walk) all wait here simultaneously — only
        // the one whose pano matches the panorama's current state when the
        // button reappears will actually proceed.
        const buttonAppeared = await waitForGuessButton(PANO_BUTTON_WAIT_MS);
        if (!buttonAppeared) {
            console.log(`[aimy] no guess button for ${panoID.slice(0,8)}… within `
                      + `${PANO_BUTTON_WAIT_MS/1000}s — abandoning`);
            return;
        }

        // Check the panorama is STILL showing this pano. If it's moved on
        // (a later setPano fired and the panorama is parked elsewhere),
        // this call is stale — abandon to avoid predicting on the wrong
        // location. The panorama's current pano is the canonical "what the
        // user sees right now".
        const currentPano = getCurrentPanoFromPanorama();
        if (currentPano && currentPano !== panoID) {
            console.log(`[aimy] panorama moved (${panoID.slice(0,8)}… → `
                      + `${currentPano.slice(0,8)}…) — abandoning stale`);
            return;
        }

        try {
            const data = await getCoordinates(panoID);
            console.log(`prediction: ${data.lat}, ${data.lng}`,
                data.country ? `(${data.country})` : '');
            showPrediction(data, roundNumber, panoID);
            if (_autoSubmit) {
                autoGuessWithRetry(data.lat, data.lng, panoID).catch(e =>
                    console.warn('[aimy] autoGuess threw:', e));
            }
        } catch (e) {
            console.warn('predict failed:', e.message || e);
        }
        roundNumber++;
    }

    // Retry wrapper: try autoGuess up to N times (because the Guess button
    // can take a moment to appear after the round starts). Aborts if a newer
    // panoID begins its own retry loop, so we don't double-fire on a stale pano.
    async function autoGuessWithRetry(lat, lng, panoID, maxAttempts = 6, intervalMs = 3000) {
        _currentAutoGuessPano = panoID;
        for (let i = 0; i < maxAttempts; i++) {
            if (_currentAutoGuessPano !== panoID) return false; // newer pano took over
            const ok = await autoGuess(lat, lng, /*quiet=*/i > 0);
            if (ok) return true;
            await wait(intervalMs);
        }
        if (_currentAutoGuessPano === panoID) {
            console.log(`[aimy] gave up auto-guessing for ${panoID} after ${maxAttempts} tries`);
        }
        return false;
    }

    function isMapVisible(m) {
        try {
            const d = m.getDiv();
            return !!(d && d.offsetParent !== null);
        } catch (e) { return true; }
    }

    function pickGuessMap() {
        const maps = W.__aimy_maps || [];
        // Prefer the map whose 'click' listener was registered most recently —
        // that's the live guess map for the current round. Read-only results
        // maps don't register 'click'.
        let pick = null, bestT = 0;
        for (const m of maps) {
            const t = m.__aimy_click_listener_at || 0;
            if (t > bestT && isMapVisible(m)) { bestT = t; pick = m; }
        }
        if (pick) return pick;
        // Fallback: last-registered visible map
        for (let i = maps.length - 1; i >= 0; i--) {
            if (isMapVisible(maps[i])) return maps[i];
        }
        if (maps.length) return maps[maps.length - 1];
        return findMapInDom();
    }

    async function autoGuess(lat, lng, quiet = false) {
        if (!W.google || !W.google.maps) {
            if (!quiet) console.warn('[aimy] google.maps not yet loaded — skipping autoguess');
            return false;
        }
        const map = pickGuessMap();
        if (!map) {
            if (!quiet) console.warn('[aimy] no Map captured AND DOM fallback found nothing — manual guess required');
            return false;
        }
        try {
            const latLng = new W.google.maps.LatLng(lat, lng);
            map.panTo(latLng);
            await wait(200);
            W.google.maps.event.trigger(map, 'click', { latLng });
        } catch (e) {
            if (!quiet) console.warn('[aimy] map click trigger failed:', e);
            return false;
        }
        await wait(700);
        const btn = findGuessButton();
        if (!btn) {
            if (!quiet) {
                const visible = Array.from(document.querySelectorAll('button'))
                    .filter(b => b.offsetParent !== null)
                    .slice(0, 12)
                    .map(b => ({
                        text: (b.textContent || '').trim().slice(0, 40),
                        disabled: b.disabled,
                        qa: b.dataset && b.dataset.qa,
                        testId: (b.dataset && (b.dataset.testid || b.dataset.testId)) || null,
                        aria: b.getAttribute('aria-label'),
                        cls: (b.className || '').toString().slice(0, 60),
                    }));
                console.warn('[aimy] guess button not found (will retry). Visible enabled buttons:', visible);
            }
            return false;
        }
        btn.click();
        console.log(`[aimy] auto-guessed at ${lat.toFixed(5)}, ${lng.toFixed(5)}`);
        return true;
    }

    function findGuessButton() {
        // Selectors in order of specificity (more specific first to avoid
        // matching the wrong button — there's also a "Place pin" before the
        // pin is placed that we DON'T want to click).
        const sels = [
            'button[data-qa="perform-guess"]',
            'button[data-qa="submit-guess"]',
            'button[data-testid="perform-guess"]',
            'button[data-test-id="game-guess-confirm"]',
            'button[data-test-id*="guess"]',
            'button[data-qa*="guess"]',
            'button[data-testid*="guess"]',
            'button[aria-label*="uess"]',
            'button[class*="guess"]',
        ];
        for (const sel of sels) {
            for (const b of document.querySelectorAll(sel)) {
                if (b && !b.disabled && b.offsetParent !== null) return b;
            }
        }
        // Text-match fallback — only enabled, visible buttons.
        const candidates = Array.from(document.querySelectorAll('button'))
            .filter(b => !b.disabled && b.offsetParent !== null);
        const patterns = [
            /^make .* guess$/i,
            /^guess$/i,
            /^place .* guess$/i,
            /^confirm .* guess$/i,
            /^submit .* guess$/i,
        ];
        for (const p of patterns) {
            const hit = candidates.find(b => p.test((b.textContent || '').trim()));
            if (hit) return hit;
        }
        // Last resort: any enabled button whose text *ends with* "guess".
        return candidates.find(b => /\bguess\b/i.test((b.textContent || '').trim())) || null;
    }

    // GM_xmlhttpRequest wrapper bypassing Mixed Content (HTTP from HTTPS) and
    // CORS — Tampermonkey privileged channel. Returns the response as a Blob.
    function gmFetchBlob(url) {
        const gm = (typeof GM_xmlhttpRequest !== 'undefined')
            ? GM_xmlhttpRequest
            : (GM && GM.xmlHttpRequest);
        return new Promise((resolve, reject) => {
            gm({
                method: 'GET',
                url,
                responseType: 'blob',
                onload: r => (r.status >= 200 && r.status < 300)
                    ? resolve(r.response)
                    : reject(new Error(`HTTP ${r.status}`)),
                onerror: e => reject(e),
                ontimeout: () => reject(new Error('timeout')),
            });
        });
    }

    // POST a multipart/form-data body via GM_xmlhttpRequest. Returns parsed JSON.
    function gmPostFormJson(url, formData) {
        const gm = (typeof GM_xmlhttpRequest !== 'undefined')
            ? GM_xmlhttpRequest
            : (GM && GM.xmlHttpRequest);
        return new Promise((resolve, reject) => {
            gm({
                method: 'POST',
                url,
                data: formData,
                responseType: 'json',
                onload: r => {
                    if (r.status < 200 || r.status >= 300) {
                        return reject(new Error(`HTTP ${r.status}: ${r.responseText}`));
                    }
                    if (r.response && typeof r.response === 'object') return resolve(r.response);
                    try { resolve(JSON.parse(r.responseText)); }
                    catch (e) { reject(e); }
                },
                onerror: e => reject(e),
                ontimeout: () => reject(new Error('timeout')),
            });
        });
    }

    // POST the panoID to our /api/v1/predict endpoint and let the backend
    // do the tile-fetch + stitch + crop + predict (same code path as
    // data_scraper.py — server-side, full browser-style headers, single
    // round trip). Browser-side tile fetching had per-pano-type schema
    // issues; the server's been hammering Google successfully at 1.36/s
    // with this exact request shape, so it's the proven path.
    function gmPostJson(url, payload) {
        const gm = (typeof GM_xmlhttpRequest !== 'undefined')
            ? GM_xmlhttpRequest
            : (GM && GM.xmlHttpRequest);
        return new Promise((resolve, reject) => {
            gm({
                method: 'POST',
                url,
                data: JSON.stringify(payload),
                headers: { 'Content-Type': 'application/json' },
                responseType: 'json',
                onload: r => {
                    if (r.status < 200 || r.status >= 300) {
                        return reject(new Error(`HTTP ${r.status}: ${r.responseText}`));
                    }
                    if (r.response && typeof r.response === 'object') return resolve(r.response);
                    try { resolve(JSON.parse(r.responseText)); }
                    catch (e) { reject(e); }
                },
                onerror: e => reject(e),
                ontimeout: () => reject(new Error('timeout')),
            });
        });
    }

    // POST returning binary (the /explain heatmap PNG). Resolves to the
    // Blob plus the raw response-header string (for the X-Explain-* metadata).
    function gmPostBlob(url, payload) {
        const gm = (typeof GM_xmlhttpRequest !== 'undefined')
            ? GM_xmlhttpRequest
            : (GM && GM.xmlHttpRequest);
        return new Promise((resolve, reject) => {
            gm({
                method: 'POST',
                url,
                data: JSON.stringify(payload),
                headers: { 'Content-Type': 'application/json' },
                responseType: 'blob',
                onload: r => {
                    if (r.status < 200 || r.status >= 300) {
                        return reject(new Error(`HTTP ${r.status}`));
                    }
                    resolve({ blob: r.response, headers: r.responseHeaders || '' });
                },
                onerror: e => reject(e),
                ontimeout: () => reject(new Error('timeout')),
            });
        });
    }

    // Streaming POST. Tampermonkey's GM_xmlhttpRequest *sometimes* delivers
    // responseText incrementally via onprogress, but on many TM versions it
    // buffers the full response and only fires onload at the end. We code
    // for both: drain() runs from both onprogress (if it fires) and onload,
    // and the cursor lets us re-parse safely without duplicate events.
    //
    // _progress_fired exposes "did we get any incremental data?" to callers
    // so the UI can fall back to a generic elapsed-time counter when TM
    // isn't streaming.
    function gmPostStreaming(url, payload, onEvent) {
        const gm = (typeof GM_xmlhttpRequest !== 'undefined')
            ? GM_xmlhttpRequest
            : (GM && GM.xmlHttpRequest);
        const state = { progressFired: false };
        const p = new Promise((resolve, reject) => {
            let cursor = 0;
            let finalData = null;
            let backendError = null;
            function drain(text) {
                const lines = text.slice(cursor).split('\n');
                cursor += lines.slice(0, -1).reduce((n, ln) => n + ln.length + 1, 0);
                for (let i = 0; i < lines.length - 1; i++) {
                    const ln = lines[i].trim();
                    if (!ln) continue;
                    try {
                        const obj = JSON.parse(ln);
                        try { onEvent(obj); } catch (e) {
                            console.warn('[aimy/stream] onEvent threw:', e);
                        }
                        if (obj.event === 'done') finalData = obj.data;
                        if (obj.event === 'error') backendError = obj.data;
                        if (obj.event === 'emit_error') console.warn(
                            '[aimy/stream] backend emit() failed:', obj.data);
                    } catch (e) {
                        console.warn('[aimy/stream] bad NDJSON line:', ln, e);
                    }
                }
            }
            gm({
                method: 'POST',
                url,
                data: JSON.stringify(payload),
                headers: { 'Content-Type': 'application/json' },
                responseType: 'text',
                onprogress: r => {
                    if (r && r.responseText && r.responseText.length > cursor) {
                        state.progressFired = true;
                        drain(r.responseText);
                    }
                },
                onload: r => {
                    if (r.status < 200 || r.status >= 300) {
                        return reject(new Error(`HTTP ${r.status}: ${r.responseText}`));
                    }
                    drain((r.responseText || '') + '\n');
                    console.log('[aimy/stream] onload — onprogress fired:',
                                state.progressFired,
                                'bytes:', (r.responseText || '').length);
                    if (finalData) resolve(finalData);
                    else if (backendError) reject(new Error(
                        'backend pipeline error: ' + (backendError.error || JSON.stringify(backendError))));
                    else reject(new Error('stream ended without done event'));
                },
                onerror: e => reject(e),
                ontimeout: () => reject(new Error('timeout')),
            });
        });
        p.__state = state;
        return p;
    }

    function setS2Status(text) {
        // Live status line shown while a refined cascade is running.
        const el = document.getElementById('aimy-s2-status');
        if (!el) return;
        if (!text) { el.style.display = 'none'; el.textContent = ''; return; }
        el.textContent = text;
        el.style.display = '';
    }

    function s2StatusForEvent(event, data) {
        // Translate raw NDJSON events into user-facing status strings.
        // Returns null for events that should be silent (just keep-alive).
        switch (event) {
            case 'heartbeat':           return null;   // keep-alive only
            case 'fetch_pano_start':    return '· fetching pano…';
            case 'fetch_pano_done':     return `· pano ${data.width}×${data.height} ready`;
            case 'stage1_start':        return '· Stage 1 inference…';
            case 'stage1_done':         return `· Stage 1: ${data.country || '?'} → starting Stage 2…`;
            case 'stage2_extract_start':  return '· Stage 2: OCR + langid…';
            case 'stage2_extract_done':   return data.script
                ? `· Stage 2: detected ${data.script} text — VLM reasoning…`
                : '· Stage 2: no readable text — keeping Stage 1';
            case 'stage2_pinpoint_start': return '· Stage 2: VLM warming up…';
            case 'stage2_pinpoint_token_start': return '· Stage 2: VLM thinking…';
            case 'stage2_pinpoint_token': {
                // Live token-count + last-80-chars snippet. The snippet is
                // mostly the model's reasoning trace; gives a real sense of
                // progress instead of a blank wait.
                const total = (data.content_chars || 0) + (data.thinking_chars || 0);
                const snippet = (data.latest || '').replace(/\s+/g, ' ').trim();
                return `· VLM thinking (${total} chars): ${snippet || '…'}`;
            }
            case 'stage2_pinpoint_token_done': {
                const total = (data.content_chars || 0) + (data.thinking_chars || 0);
                return `· VLM done thinking (${total} chars) — parsing…`;
            }
            case 'stage2_pinpoint_done':  return data.precision === 'country'
                ? '· Stage 2: VLM declined to narrow — keeping Stage 1'
                : `· Stage 2: VLM → ${data.precision} "${data.queryable || '?'}"`;
            case 'stage2_pinpoint_timeout': return '· Stage 2: VLM timed out — falling back to Stage 1';
            case 'stage2_geocode_start':  return '· Stage 2: geocoding…';
            case 'stage2_geocode_done':   return data.hit
                ? `· Stage 2: matched "${data.hit.split(',')[0]}"`
                : '· Stage 2: geocoder missed — falling back to Stage 1';
            case 'stage2_done':         return '';   // about to be replaced by `done`
            case 'done':                return '';
            case 'error':               return `· error: ${data.error || 'unknown'}`;
            default:                    return '';
        }
    }

    async function getCoordinates(panoID) {
        // For the refined cascade we use the streaming endpoint so we can
        // show live progress in the AIMY panel (Stage 2 takes 5-40 s and
        // otherwise looks frozen).
        //
        // Tampermonkey's GM_xmlhttpRequest doesn't always stream the
        // responseText via onprogress (buffering varies by version), so we
        // also run a client-side elapsed-time counter as a fallback.
        // Real events arriving via onprogress override the counter; if no
        // events ever arrive (buffered TM), the counter at least shows
        // "yes the request is still going" with the wall-clock elapsed.
        let data;
        if (_cascade === 'refined') {
            const t0 = Date.now();
            let realEventSeen = false;
            const elapsed = () => ((Date.now() - t0) / 1000).toFixed(0);
            setS2Status('· starting…');
            const counter = setInterval(() => {
                if (!realEventSeen) setS2Status(`· running (${elapsed()}s)…`);
            }, 500);
            try {
                data = await gmPostStreaming(
                    PREDICT_STREAM_URL,
                    { panoID, cascade: _cascade },
                    (ev) => {
                        const s = s2StatusForEvent(ev.event, ev.data || {});
                        if (s) {
                            realEventSeen = true;   // events arriving → stop the counter
                            setS2Status(s);
                        }
                    },
                );
                // Diagnostic dump — surface every Stage 2 field at the
                // end of each refined call so we can see why the model
                // did/didn't refine without parsing the raw response.
                console.log('[aimy/s2]', {
                    used: data.stage2_used,
                    source: data.stage2_source,
                    precision: data.stage2_precision,
                    confidence: data.stage2_confidence,
                    queryable: data.stage2_queryable,
                    match: data.stage2_match_name && data.stage2_match_name.split(',')[0],
                    explanation: data.stage2_explanation,
                    seconds: data.stage2_seconds,
                    extract_s: data.stage2_extract_seconds,
                    pinpoint_s: data.stage2_pinpoint_seconds,
                    geocode_s: data.stage2_geocode_seconds,
                    error: data.stage2_error,
                });
            } finally {
                clearInterval(counter);
                setS2Status('');
            }
        } else {
            data = await gmPostJson(PREDICT_URL, { panoID, cascade: _cascade });
        }
        if (typeof data.lat !== 'number') data.lat = data.final_lat;
        if (typeof data.lng !== 'number') data.lng = data.final_lng;
        return data;
    }

    async function submitGuess(lat, lng, roundNumber) {
        const gameID = getGameID();
        const href = window.location.href;
        let apiURL, payload;
        if (href.includes('/team-duels/')) {
            // Team duels: gs2 host, two-token path, payload includes ISO time.
            // The session/round IDs aren't in the page URL; we sniff them from
            // the page's own gs2 traffic (see hookTeamDuelsIds at the top).
            if (!_teamDuelsCtx.sessionId || !_teamDuelsCtx.roundId) {
                throw new Error('team-duels session/round IDs not yet captured — wait for round state to load');
            }
            apiURL = `https://gs2.geoguessr.com/${_teamDuelsCtx.sessionId}/${_teamDuelsCtx.roundId}/guess`;
            payload = { lat, lng, roundNumber, time: new Date().toISOString() };
        } else if (href.includes('/battle-royale/')) {
            apiURL = `https://game-server.geoguessr.com/api/battle-royale/${gameID}/guess`;
            payload = { lat, lng, roundNumber };
        } else if (href.includes('/duels/')) {
            apiURL = `https://game-server.geoguessr.com/api/duels/${gameID}/guess`;
            payload = { lat, lng, roundNumber };
        } else if (href.includes('/game/')) {
            // Standard 5-round game — same-origin API, different payload
            apiURL = `https://www.geoguessr.com/api/v3/games/${gameID}`;
            payload = { token: gameID, lat, lng, timedOut: false };
        } else {
            console.warn(`unknown game type for ${href}, defaulting to duels`);
            apiURL = `https://game-server.geoguessr.com/api/duels/${gameID}/guess`;
            payload = { lat, lng, roundNumber };
        }

        const headers = {
            origin: "https://www.geoguessr.com",
            referer: apiURL,
            "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
            "sec-ch-ua": '"Chromium";v="106", "Google Chrome";v="106", "Not;A=Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "macOS",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36",
            "x-client": "web",
            "Content-Type": "application/json",
        };
        await wait(5000); // beginning of round
        for (let i = 0; i < 5; i++) {
            try {
                const response = await fetch(apiURL, {
                    method: "POST",
                    credentials: "include",
                    headers: headers,
                    body: JSON.stringify(payload),
                });
                if (!response.ok) {
                    throw new Error(`Guess submission failed: ${await response.text()}`);
                }
                return { resp: response, body: await response.json() };
            } catch (error) {
                console.error("Error submitting guess:", error);
                await wait(1000);
                throw error;
            }
        }
    }

    var originalOpen = XMLHttpRequest.prototype.open;
    XMLHttpRequest.prototype.open = function(method, url) {
        if (method.toUpperCase() === 'POST' &&
            (url.startsWith('https://maps.googleapis.com/$rpc/google.internal.maps.mapsjs.v1.MapsJsInternalService/GetMetadata') ||
             url.startsWith('https://maps.googleapis.com/$rpc/google.internal.maps.mapsjs.v1.MapsJsInternalService/SingleImageSearch'))) {

            this.addEventListener('load', async function () {
                let jsonResponse;
                try { jsonResponse = JSON.parse(this.responseText); }
                catch (e) { return; }  // not JSON — different RPC sharing the URL prefix

                // Custom-map metadata responses can come back with a different
                // shape (e.g. an error envelope, or a Single­ImageSearch hit
                // without panoID); guard the access so we don't spam console.
                let panoID;
                try { panoID = jsonResponse[1][0][1][1]; } catch (e) { return; }
                if (typeof panoID !== 'string' || !panoID) return;

                onPanoIDDetected(panoID).catch(e =>
                    console.warn('[aimy] onPanoIDDetected (xhr) threw:', e));
            });
        }
        return originalOpen.apply(this, arguments);
    };

    var originalFetch = window.fetch;
    window.fetch = function() {
        return originalFetch.apply(this, arguments).then(function(response) {
            if (response.url.includes('/games/') && response.url.includes('/round/')) {
                if (isNewGame()) {
                    roundNumber = 1;
                    console.log('New game started. Round number reset to 1.');
                }
            }
            return response;
        });
    };

    // Show the overlay on page load (collapsed if that was the last state) so
    // the controls (mode, auto toggle) are reachable before round 1.
    function bootOverlay() {
        try { ensureOverlay(); }
        catch (e) { console.warn('[aimy] overlay boot:', e); }
    }
    if (document.body) bootOverlay();
    else document.addEventListener('DOMContentLoaded', bootOverlay);
})();
