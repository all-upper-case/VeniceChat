document.addEventListener('DOMContentLoaded', () => {
    let chatHistory = [];
    let summaries = [];
    let currentVisualMemory = "";
    let currentSelfMemory = "";
    let memoryLogs = [];
    let isGenerating = false;
    let guidedModeEnabled = false;
    let activeEditIndex = -1;
    let activeEditType = 'message';
    let runningInput = 0;
    let runningOutput = 0;
    let diffModeMsgs = new Set();
    
    // Audit Mode State
    let isAuditMode = false;
    let auditContextSelection = { batches: [], includeRaw: true };
    let parentSummariesCache = [];

    // Arena Vars
    window.isArenaMode = false;
    window.arenaData = null;
    window.activeArenaTab = null;

    const chatbox = document.getElementById('chatbox');
    const userInput = document.getElementById('userInput');
    const modal = document.getElementById('settingsModal');
    const editOverlay = document.getElementById('edit-overlay');
    const editTextarea = document.getElementById('edit-textarea');
    const guidedModal = document.getElementById('guided-modal');
    const guidedInput = document.getElementById('guided-input');
    const refineGuidanceModal = document.getElementById('refine-guidance-modal');
    const refineGuidanceInput = document.getElementById('refine-guidance-input');
    const vmTextarea = document.getElementById('visual-memory-text');
    const analyticsModal = document.getElementById('analytics-modal');
    const loreExtractorModal = document.getElementById('lore-extractor-modal');
    const apiExplorerModal = document.getElementById('api-explorer-modal');

    // Model Selector Enhancements
    const modelSearchInput = document.getElementById('model-search-input');
    const modelFilterTags = document.querySelectorAll('.filter-tag');

    if (modelSearchInput) {
        modelSearchInput.addEventListener('input', (e) => filterAllModelSelectors());
    }

    const modelSortSelect = document.getElementById('model-sort-select');
    if (modelSortSelect) {
        modelSortSelect.addEventListener('change', () => filterAllModelSelectors());
    }

    modelFilterTags.forEach(tag => {
        tag.addEventListener('click', () => {
            modelFilterTags.forEach(t => t.classList.remove('active'));
            tag.classList.add('active');
            filterAllModelSelectors();
        });
    });

    function parsePrice(pricingStr, type) {
        // pricing format example: "$0.56 / $3.50 ($0.11 Cache)"
        if (!pricingStr || pricingStr === 'N/A') return 0;
        const parts = pricingStr.split('/');
        if (type === 'input') return parseFloat(parts[0].replace(/[^0-9.]/g, '')) || 0;
        if (type === 'output') return parts[1] ? (parseFloat(parts[1].split('(')[0].replace(/[^0-9.]/g, '')) || 0) : 0;
        if (type === 'cache') {
            const cacheMatch = pricingStr.match(/\$(\d+\.?\d*)\s+Cache/);
            return cacheMatch ? parseFloat(cacheMatch[1]) : 0;
        }
        return 0;
    }

    function parseContext(contextStr) {
        if (!contextStr || contextStr === 'N/A') return 0;
        return parseInt(contextStr.replace(/[^0-9]/g, '')) || 0;
    }

    function filterAllModelSelectors() {
        const searchInput = document.getElementById('model-search-input');
        const query = searchInput ? searchInput.value : "";
        const trait = document.querySelector('.filter-tag.active')?.dataset.filter || 'all';
        const sort = document.getElementById('model-sort-select')?.value || 'name';
        
        const selectors = document.querySelectorAll('.model-selector-sync');
        selectors.forEach(select => {
            if (availableModels) {
                const currentVal = select.dataset.currentValue || select.value;
                populateFilteredSelect(select, query, trait, sort, false, false);
                if (currentVal && currentVal !== "Loading models...") {
                    select.value = currentVal;
                }
            } else {
                select.innerHTML = '<option value="">Loading models...</option>';
            }
        });
    }

    function populateFilteredSelect(select, query = "", trait = "all", sort = "name", forceAll = false, showDefault = false) {
        if (!select || !availableModels) return;

        const lowQuery = forceAll ? "" : query.toLowerCase();
        const effectiveTrait = forceAll ? "all" : trait;
        const currentVal = select.dataset.currentValue || select.value;
        
        select.innerHTML = '';
        if (showDefault) {
            const defOpt = document.createElement('option');
            defOpt.value = "";
            defOpt.textContent = "(Default)";
            select.appendChild(defOpt);
        }

        let allFiltered = [];

        for (const group in availableModels) {
            let filteredGroup = availableModels[group].filter(m => {
                const matchesQuery = m.name.toLowerCase().includes(lowQuery) || m.id.toLowerCase().includes(lowQuery);
                let matchesTrait = true;
                if (effectiveTrait === 'vision') matchesTrait = m.vision;
                else if (effectiveTrait === 'reasoning') matchesTrait = m.traits?.toLowerCase().includes('reasoning') || m.id.includes('reasoning') || m.id.includes('thinking');
                else if (effectiveTrait === 'private') matchesTrait = m.traits?.toLowerCase().includes('private');
                else if (effectiveTrait === 'beta') matchesTrait = (m.tags && m.tags.includes('BETA')) || m.id.includes('beta');

                return matchesQuery && matchesTrait;
            });

            if (sort === 'name' && filteredGroup.length > 0) {
                filteredGroup.sort((a, b) => a.name.localeCompare(b.name));
                const optgroup = document.createElement('optgroup');
                optgroup.label = group;
                filteredGroup.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m.id;
                    opt.textContent = (m.vision ? '👁️ ' : '') + m.name;
                    if (m.id === currentVal) opt.selected = true;
                    optgroup.appendChild(opt);
                });
                select.appendChild(optgroup);
            } else if (sort !== 'name' && filteredGroup.length > 0) {
                allFiltered.push(...filteredGroup);
            }
        }

        if (sort !== 'name' && allFiltered.length > 0) {
            allFiltered.sort((a, b) => {
                if (sort === 'context') return parseContext(b.context) - parseContext(a.context);
                if (sort === 'input') return parsePrice(a.pricing, 'input') - parsePrice(b.pricing, 'input');
                if (sort === 'output') return parsePrice(a.pricing, 'output') - parsePrice(b.pricing, 'output');
                if (sort === 'cache') return parsePrice(a.pricing, 'cache') - parsePrice(b.pricing, 'cache');
                return 0;
            });

            allFiltered.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m.id;
                opt.textContent = (m.vision ? '👁️ ' : '') + m.name;
                if (m.id === currentVal) opt.selected = true;
                select.appendChild(opt);
            });
        }
    }

    // Architect UI Vars
    const architectModal = document.getElementById('architect-modal');
    const architectDisplay = document.getElementById('architect-chat-display');
    const architectInput = document.getElementById('architect-input');
    let architectHistory = [];

    // Helper to safely bind events so one missing ID doesn't crash the whole UI
    const safeBind = (id, event, handler) => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener(event, handler);
        } else {
            console.warn(`SafeBind Warning: Element '${id}' not found for event '${event}'.`);
        }
    };

    const autoExpandTextarea = (ta) => {
        if (!ta) return;
        ta.style.height = 'auto';
        ta.style.height = ta.scrollHeight + 'px';
    };

    function renderDiff(oldStr, newStr) {
        const a = oldStr.split(/(\s+)/);
        const b = newStr.split(/(\s+)/);
        const matrix = Array(a.length + 1).fill(null).map(() => Array(b.length + 1).fill(0));
        for (let i = 1; i <= a.length; i++) {
            for (let j = 1; j <= b.length; j++) {
                if (a[i - 1] === b[j - 1]) {
                    matrix[i][j] = matrix[i - 1][j - 1] + 1;
                } else {
                    matrix[i][j] = Math.max(matrix[i - 1][j], matrix[i][j - 1]);
                }
            }
        }
        let i = a.length, j = b.length;
        const diff = [];
        while (i > 0 || j > 0) {
            if (i > 0 && j > 0 && a[i - 1] === b[j - 1]) {
                diff.unshift({ type: 'equal', val: a[i - 1] });
                i--; j--;
            } else if (j > 0 && (i === 0 || matrix[i][j - 1] >= matrix[i - 1][j])) {
                diff.unshift({ type: 'add', val: b[j - 1] });
                j--;
            } else if (i > 0 && (j === 0 || matrix[i][j - 1] < matrix[i - 1][j])) {
                diff.unshift({ type: 'remove', val: a[i - 1] });
                i--;
            }
        }

        let html = "";
        diff.forEach(part => {
            const escapeHtml = (text) => text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
            const safeVal = escapeHtml(part.val);

            if (part.type === 'equal') {
                html += safeVal;
            } else if (part.type === 'add') {
                if (part.val.trim().length > 0) html += `<span class="diff-add">${safeVal}</span>`;
                else html += safeVal;
            } else if (part.type === 'remove') {
                if (part.val.trim().length > 0) html += `<del class="diff-remove">${safeVal}</del>`;
                else html += safeVal;
            }
        });
        return html;
    }

    async function handleUndoRefine(index) {
        if (!confirm("Revert this message back to its original pre-refined state?")) return;
        try {
            const res = await fetch('/undo_ai_refine', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({index: index})
            });
            const d = await res.json();
            if (d.success) {
                diffModeMsgs.delete(index);
                await loadHistory();
            } else {
                alert("Undo Error: " + d.error);
            }
        } catch(e) { alert("Failed to undo."); }
    }

    // Initialization
    let attachedImageBase64 = null;
    let availableModels = null;

    function updateBalanceDisplay(balance) {
        if (balance === undefined || balance === null) return;
        const container = document.getElementById('account-balance-display');
        const amount = document.getElementById('balance-amount');
        if (container && amount) {
            container.style.display = 'block';
            amount.textContent = `${parseFloat(balance).toFixed(4)}`;
        }
    }

    function applyInterfaceSettings(fontSize, bgColor) {
        document.documentElement.style.setProperty('--font-size', `${fontSize}px`);
        document.documentElement.style.setProperty('--bg-dark', bgColor);
    }

    const init = async () => {
        try {
            // Concurrent fetching for speed
            const [mRes, settingsRes, bRes] = await Promise.allSettled([
                fetch('/venice_models'),
                fetch('/get_settings'),
                fetch('/get_balance')
            ]);

            if (mRes.status === 'fulfilled' && mRes.value.ok) {
                availableModels = await mRes.value.json();
                filterAllModelSelectors();
            }

            if (settingsRes.status === 'fulfilled' && settingsRes.value.ok) {
                const d = await settingsRes.value.json();
                if(d.interface) applyInterfaceSettings(d.interface.font_size, d.interface.bg_color);
                if(d.tts) window.ttsSettings = d.tts;
                
                // Sync summarizer settings to UI
                if (d.summarizer) {
                    const ids = {'live-sum-enabled': 'live_summary_enabled', 'sum-keep': 'recent_turns_to_keep', 'sum-batch': 'batch_size'};
                    for (const [elId, key] of Object.entries(ids)) {
                        const el = document.getElementById(elId);
                        if (el) el[el.type === 'checkbox' ? 'checked' : 'value'] = d.summarizer[key];
                    }
                }
            }

            if (bRes.status === 'fulfilled' && bRes.value.ok) {
                const bData = await bRes.value.json();
                if (bData?.balance) updateBalanceDisplay(bData.balance);
            }

            // Load UI data
            await Promise.all([loadHistory(), loadSidebar()]);
            
            updateAttachmentButtonVisibility();
            checkScanStatus();

            // Post-init cleanup
            document.querySelectorAll('textarea').forEach(ta => {
                autoExpandTextarea(ta);
                ta.addEventListener('input', () => autoExpandTextarea(ta));
            });
        } catch (e) {
            console.error("Initialization failed:", e);
        }
    };

    init();

    // --- TTS LOGIC ---
    let ttsCache = {}; // Map of contentHash -> blobUrl
    let currentTTS = {
        msgIndex: -1,
        contentHash: null,
        audio: null
    };

    function getContentHash(text) {
        // Simple string hash for client-side cache keying
        let hash = 0;
        for (let i = 0; i < text.length; i++) {
            const char = text.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash |= 0;
        }
        return hash.toString();
    }

    function updateTTSButtonStates() {
        const allBtns = document.querySelectorAll('.tts-btn');
        allBtns.forEach(btn => {
            const idx = parseInt(btn.dataset.index);
            const msg = chatHistory[idx];
            if (!msg) return;
            const textContent = Array.isArray(msg.content) ? (msg.content.find(p => p.type === 'text')?.text || "") : msg.content;
            const hash = getContentHash(textContent);

            if (ttsCache[hash]) {
                btn.classList.add('tts-ready');
                if (currentTTS.msgIndex === idx && currentTTS.contentHash === hash) {
                    if (currentTTS.audio && !currentTTS.audio.paused) {
                        btn.innerHTML = '⏹ Stop';
                    } else {
                        btn.innerHTML = '▶ Play';
                    }
                } else {
                    btn.innerHTML = '▶ Play';
                }
            }
        });
    }

    function renderTTSControls(index) {
        const targetDiv = document.querySelector(`#msg-${index} .tools-panel`);
        if (!targetDiv) return;

        // Remove any existing controls
        const existing = targetDiv.querySelector('.tts-playback-controls');
        if (existing) existing.remove();

        const msg = chatHistory[index];
        const textContent = Array.isArray(msg.content) ? (msg.content.find(p => p.type === 'text')?.text || "") : msg.content;
        const hash = getContentHash(textContent);

        if (currentTTS.msgIndex !== index || currentTTS.contentHash !== hash || !ttsCache[hash]) return;

        const controls = document.createElement('div');
        controls.className = 'tts-playback-controls';
        controls.innerHTML = `
            <button class="tts-ctrl-btn" data-skip="-15" title="Back 15s">-15s</button>
            <button class="tts-ctrl-btn" data-skip="-5" title="Back 5s">-5s</button>
            <button class="tts-ctrl-btn play-pause-btn" title="Play/Pause">â¸</button>
            <button class="tts-ctrl-btn" data-skip="5" title="Forward 5s">+5s</button>
            <button class="tts-ctrl-btn" data-skip="15" title="Forward 15s">+15s</button>
            <div class="tts-progress-container">
                <div class="tts-progress-bar"></div>
            </div>
        `;

        controls.querySelectorAll('[data-skip]').forEach(btn => {
            btn.onclick = (e) => {
                e.stopPropagation();
                if (currentTTS.audio) {
                    currentTTS.audio.currentTime += parseFloat(btn.dataset.skip);
                }
            };
        });

        const ppBtn = controls.querySelector('.play-pause-btn');
        ppBtn.onclick = (e) => {
            e.stopPropagation();
            if (!currentTTS.audio) return;
            if (currentTTS.audio.paused) {
                currentTTS.audio.play();
                ppBtn.textContent = 'â¸';
            } else {
                currentTTS.audio.pause();
                ppBtn.textContent = '▶';
            }
            updateTTSButtonStates();
        };

        targetDiv.appendChild(controls);

        // Progress Tracking
        if (currentTTS.audio) {
            const bar = controls.querySelector('.tts-progress-bar');
            currentTTS.audio.ontimeupdate = () => {
                const pct = (currentTTS.audio.currentTime / currentTTS.audio.duration) * 100;
                if (bar) bar.style.width = pct + '%';
            };
            ppBtn.textContent = currentTTS.audio.paused ? '▶' : 'â¸';
        }
    }

    let ttsRefAudioBase64 = null;

    safeBind('tts-ref-audio', 'change', (e) => {
        const file = e.target.files[0];
        if (!file) {
            ttsRefAudioBase64 = null;
            document.getElementById('tts-clone-status').style.display = 'none';
            document.getElementById('clear-tts-clone-btn').style.display = 'none';
            return;
        }
        const reader = new FileReader();
        reader.onload = (event) => {
            // Mistral expects the Base64 data without the Data URL prefix
            const b64 = event.target.result.split(',')[1];
            ttsRefAudioBase64 = b64;
            const status = document.getElementById('tts-clone-status');
            if (status) status.style.display = 'block';
            const clearBtn = document.getElementById('clear-tts-clone-btn');
            if (clearBtn) clearBtn.style.display = 'block';
        };
        reader.readAsDataURL(file);
    });

    safeBind('clear-tts-clone-btn', 'click', () => {
        ttsRefAudioBase64 = null;
        document.getElementById('tts-ref-audio').value = '';
        document.getElementById('tts-clone-status').style.display = 'none';
        document.getElementById('clear-tts-clone-btn').style.display = 'none';
    });

    safeBind('set-tts-model', 'change', (e) => {
        const cloneContainer = document.getElementById('mistral-clone-container');
        if (cloneContainer) {
            cloneContainer.style.display = e.target.value.includes('voxtral') ? 'block' : 'none';
        }
    });

    async function handleTTS(index) {
        const msg = chatHistory[index];
        if (!msg || !msg.content) return;

        const textContent = Array.isArray(msg.content) ? (msg.content.find(p => p.type === 'text')?.text || "") : msg.content;
        const hash = getContentHash(textContent);
        const btn = document.querySelector(`#msg-${index} .tts-btn`);
        if (!btn) return;

        // If switching messages or content changed at same index, pause current
        if ((currentTTS.msgIndex !== index || currentTTS.contentHash !== hash) && currentTTS.audio) {
            currentTTS.audio.pause();
        }

        // 1. GENERATION PHASE (If not in cache)
        if (!ttsCache[hash]) {
            btn.innerHTML = '⌛ Generating...';
            btn.disabled = true;

            try {
                const res = await fetch('/tts', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        text: textContent,
                        ref_audio: ttsRefAudioBase64 // Pass the cloned voice if loaded
                    })
                });

                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.error || "Generation failed");
                }

                const d = await res.json();
                if (d.balance) updateBalanceDisplay(d.balance);

                ttsCache[hash] = d.url;
                btn.classList.add('tts-ready');
            } catch (e) {
                console.error("TTS Error:", e);
                alert("TTS Error: " + e.message);
                btn.innerHTML = '❌ Error';
                return;
            } finally {
                btn.disabled = false;
                updateTTSButtonStates();
            }
        }

        // 2. PLAYBACK PHASE
        if (currentTTS.msgIndex !== index || currentTTS.contentHash !== hash) {
            currentTTS.msgIndex = index;
            currentTTS.contentHash = hash;
            if (currentTTS.audio) {
                currentTTS.audio.pause();
                currentTTS.audio.onended = null;
                currentTTS.audio.ontimeupdate = null;
            }

            // Create a brand new Audio object to ensure fresh duration detection
            const audioObj = new Audio(ttsCache[hash]);
            audioObj.preload = "auto";
            currentTTS.audio = audioObj;

            currentTTS.audio.onended = () => {
                updateTTSButtonStates();
                renderTTSControls(index);
            };

            // Critical: Wait for metadata so we have a valid duration for seeking
            currentTTS.audio.addEventListener('loadedmetadata', () => {
                renderTTSControls(index);
                updateTTSButtonStates();
            });

            // Fallback for immediate UI update
            renderTTSControls(index);
        }

        if (currentTTS.audio) {
            if (!currentTTS.audio.paused) {
                currentTTS.audio.pause();
            } else {
                currentTTS.audio.play().catch(e => {
                    console.error("Playback Blocked:", e);
                    alert("Playback blocked by browser. Click Play again.");
                });
            }
        }

        updateTTSButtonStates();
        renderTTSControls(index);
    }

    // --- ARCHITECT LOGIC ---
    safeBind('open-architect-btn', 'click', () => {
        if (architectModal) architectModal.style.display = 'block';
        if (architectHistory.length === 0 && architectDisplay) {
            architectHistory.push({role:'assistant', content: architectDisplay.firstElementChild.textContent});
        }
        if (architectInput) architectInput.focus();
    });

    if (architectInput) {
        architectInput.addEventListener('keydown', (e) => {
            if(e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                const start = architectInput.selectionStart;
                const end = architectInput.selectionEnd;
                const val = architectInput.value;
                architectInput.value = val.substring(0, start) + "\n" + val.substring(end);
                architectInput.selectionStart = architectInput.selectionEnd = start + 1;
            }
        });
    }

    safeBind('architect-send-btn', 'click', async () => {
        if (!architectInput || !architectDisplay) return;
        const txt = architectInput.value.trim();
        if(!txt) return;
        architectInput.value = '';
        autoExpandTextarea(architectInput);

        const uDiv = document.createElement('div'); uDiv.className = 'message user'; uDiv.innerText = txt;
        architectDisplay.appendChild(uDiv);
        architectHistory.push({role:'user', content:txt});
        architectDisplay.scrollTop = architectDisplay.scrollHeight;

        const aDiv = document.createElement('div'); aDiv.className = 'message assistant'; aDiv.innerText = '...';
        architectDisplay.appendChild(aDiv);
        architectDisplay.scrollTop = architectDisplay.scrollHeight;

        let fullContent = "";
        try {
            const res = await fetch('/architect_chat', {
                method: 'POST', headers: {'Content-Type':'application/json'},
                body: JSON.stringify({history: architectHistory})
            });
            const reader = res.body.getReader();
            const dec = new TextDecoder();

            while(true) {
                const {done, value} = await reader.read();
                if(done) break;
                const chunk = dec.decode(value);
                chunk.split('\n\n').forEach(l => {
                    if(l.startsWith('data: ')) {
                        try { 
                            const d = JSON.parse(l.substring(6)); 
                            if(d.content) fullContent += d.content; 
                            if(d.balance) updateBalanceDisplay(d.balance);
                        } catch(e){}
                    }
                });
                aDiv.innerText = fullContent;
                architectDisplay.scrollTop = architectDisplay.scrollHeight;
            }

            if(fullContent.includes('[[SCENARIO_START]]')) {
                const parts = fullContent.split('[[SCENARIO_START]]');
                const finalPrompt = parts[1].trim();
                aDiv.innerHTML += `<br><br><b style="color:var(--accent)">Creating Scenario...</b>`;
                await createScenario(finalPrompt);
            } else {
                architectHistory.push({role:'assistant', content: fullContent});
            }

        } catch(e) { aDiv.innerText += " [Error]"; }
    });

    async function createScenario(promptText) {
        try {
            const res = await fetch('/create_scenario_chat', {
                method: 'POST', headers: {'Content-Type':'application/json'},
                body: JSON.stringify({prompt: promptText})
            });
            const data = await res.json();
            if(data.success) {
                if (architectModal) architectModal.style.display = 'none';
                const sidebar = document.getElementById('sidebar');
                if (sidebar) sidebar.classList.add('hidden');
                architectHistory = [];
                if (architectDisplay) architectDisplay.innerHTML = '<div class="message assistant">Hello. I am the Scenario Architect. I can help you design a rich starting prompt for a new roleplay or story. What kind of genre or setting are you looking for today?</div>';
                await loadHistory();
            } else {
                alert("Error creating scenario: " + data.error);
            }
        } catch(e) { console.error(e); alert("Failed to launch scenario."); }
    }

    let activeRefineIndex = -1;

    async function handleAIRefine(index) {
        activeRefineIndex = index;
        if (refineGuidanceModal) {
            refineGuidanceModal.style.display = 'block';
            if (refineGuidanceInput) {
                refineGuidanceInput.value = '';
                refineGuidanceInput.focus();
            }
        }
    }

    let activeFindReplaceIndex = -1;
    function handleFindReplace(index) {
        activeFindReplaceIndex = index;
        const modal = document.getElementById('find-replace-modal');
        if (modal) {
            modal.style.display = 'block';
            const findIn = document.getElementById('fr-find');
            if (findIn) {
                findIn.value = '';
                findIn.focus();
            }
            const replaceIn = document.getElementById('fr-replace');
            if (replaceIn) replaceIn.value = '';
        }
    }

    safeBind('close-find-replace-x', 'click', () => {
        const modal = document.getElementById('find-replace-modal');
        if (modal) modal.style.display = 'none';
        activeFindReplaceIndex = -1;
    });

    safeBind('fr-cancel', 'click', () => {
        const modal = document.getElementById('find-replace-modal');
        if (modal) modal.style.display = 'none';
        activeFindReplaceIndex = -1;
    });

    safeBind('fr-submit', 'click', async () => {
        const findStr = document.getElementById('fr-find').value;
        const replaceStr = document.getElementById('fr-replace').value;
        if (!findStr) return;

        const index = activeFindReplaceIndex;
        const modal = document.getElementById('find-replace-modal');

        try {
            const res = await fetch('/find_replace_message', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ index: index, find: findStr, replace: replaceStr })
            });
            const d = await res.json();
            if (d.success) {
                if (modal) modal.style.display = 'none';
                activeFindReplaceIndex = -1;
                // Treat it like a refined message to show diff if desired
                diffModeMsgs.add(index);
                await loadHistory();
            } else {
                alert("Error: " + d.error);
            }
        } catch (e) {
            alert("Failed to perform replacement.");
        }
    });

    safeBind('refine-guidance-cancel', 'click', () => {
        if (refineGuidanceModal) refineGuidanceModal.style.display = 'none';
        activeRefineIndex = -1;
    });

    safeBind('refine-guidance-cancel-x', 'click', () => {
        if (refineGuidanceModal) refineGuidanceModal.style.display = 'none';
        activeRefineIndex = -1;
    });

    safeBind('refine-context-toggle', 'change', (e) => {
        const container = document.getElementById('refine-context-depth-container');
        if (container) container.style.display = e.target.checked ? 'flex' : 'none';
    });

    safeBind('refine-guidance-submit', 'click', async () => {
        const index = activeRefineIndex;
        const guidance = refineGuidanceInput ? refineGuidanceInput.value.trim() : "";
        const includeContext = document.getElementById('refine-context-toggle')?.checked || false;
        const contextDepth = parseInt(document.getElementById('refine-context-depth')?.value) || 5;

        if (refineGuidanceModal) refineGuidanceModal.style.display = 'none';

        const tools = document.querySelectorAll('.tools-panel');
        tools.forEach(t => t.style.display = 'none');

        const originalText = chatHistory[index].content;
        const targetDiv = document.getElementById(`msg-${index}`);

        const overlay = document.createElement('div');
        overlay.style.cssText = "position:absolute; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.5); display:flex; align-items:center; justify-content:center; border-radius:12px; z-index:10;";
        overlay.innerHTML = `<span style="color:white; font-weight:bold;">Refining...</span>`;
        if (targetDiv) targetDiv.appendChild(overlay);

        try {
            const res = await fetch('/ai_refine_message', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    index: index, 
                    guidance: guidance,
                    include_context: includeContext,
                    context_depth: contextDepth
                })
            });
            const d = await res.json();
            if (d.success) {
                if (d.no_changes) {
                    alert("The AI model determined that no changes were necessary.");
                } else {
                    chatHistory[index].original_content = originalText;
                    chatHistory[index].content = d.refined_text;
                    chatHistory[index].refine_logic = d.refine_logic;
                    diffModeMsgs.add(index);
                }
                renderChat();
            } else {
                alert("Refinement Error: " + d.error);
                if (overlay) overlay.remove();
            }
        } catch(e) {
            alert("Failed to refine message.");
            if (overlay) overlay.remove();
        }
        activeRefineIndex = -1;
    });

    // Character Gallery Vars
    const charactersModal = document.getElementById('characters-modal');
    const charGalleryGrid = document.getElementById('character-gallery-grid');
    const charSearchInput = document.getElementById('char-search-input');
    let allCharacters = [];

    safeBind('open-characters-btn', 'click', () => {
        if (charactersModal) charactersModal.style.display = 'block';
        if (allCharacters.length === 0) fetchCharacters();
    });

    const charEditorModal = document.getElementById('character-editor-modal');

    safeBind('open-create-persona-btn', 'click', () => {
        openCharacterEditor(null);
    });

    safeBind('close-char-editor-x', 'click', () => {
        if(charEditorModal) charEditorModal.style.display = 'none';
    });

    function openCharacterEditor(charData = null) {
        if (!charEditorModal) return;

        // Populate Model Select
        const select = document.getElementById('ce-model');
        select.innerHTML = '';
        if (availableModels) {
            for (const group in availableModels) {
                const optgroup = document.createElement('optgroup');
                optgroup.label = group;
                availableModels[group].forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m.id;
                    opt.textContent = m.name;
                    optgroup.appendChild(opt);
                });
                select.appendChild(optgroup);
            }
        }

        const title = document.getElementById('char-editor-title');
        const nameIn = document.getElementById('ce-name');
        const descIn = document.getElementById('ce-desc');
        const instIn = document.getElementById('ce-inst');
        const slugIn = document.getElementById('ce-slug');
        const webIn = document.getElementById('ce-web');
        const delBtn = document.getElementById('ce-delete-btn');
        const statusDiv = document.getElementById('ce-status');

        statusDiv.textContent = "";

        if (charData) {
            title.textContent = "Edit Persona: " + charData.name;
            nameIn.value = charData.name || "";
            descIn.value = charData.description || "";
            instIn.value = charData.instructions || "";
            select.value = charData.modelId || "venice-uncensored";
            slugIn.value = charData.slug || "";
            webIn.checked = !!charData.webEnabled;
            delBtn.style.display = 'block';
        } else {
            title.textContent = "Create New Persona";
            nameIn.value = ""; descIn.value = ""; instIn.value = ""; slugIn.value = "";
            select.value = "venice-uncensored";
            webIn.checked = false;
            delBtn.style.display = 'none';
        }

        charEditorModal.style.display = 'block';
    }

    safeBind('ce-save-btn', 'click', async () => {
        const btn = document.getElementById('ce-save-btn');
        const statusDiv = document.getElementById('ce-status');

        const payload = {
            name: document.getElementById('ce-name').value.trim(),
            description: document.getElementById('ce-desc').value.trim(),
            instructions: document.getElementById('ce-inst').value.trim(),
            modelId: document.getElementById('ce-model').value,
            webEnabled: document.getElementById('ce-web').checked,
            isPublic: false
        };

        if (!payload.name || !payload.instructions) {
            statusDiv.textContent = "Name and Instructions are required.";
            return;
        }

        const slug = document.getElementById('ce-slug').value;
        const isEdit = !!slug;

        btn.disabled = true;
        btn.textContent = "Saving...";
        statusDiv.textContent = "Sending to Venice...";

        try {
            const url = isEdit ? `/edit_character/${slug}` : `/create_character`;
            const method = isEdit ? 'PUT' : 'POST';

            const res = await fetch(url, {
                method: method,
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
            const d = await res.json();

            if (d.success) {
                statusDiv.textContent = "Saved successfully!";
                setTimeout(() => {
                    charEditorModal.style.display = 'none';
                    fetchCharacters(); // Refresh gallery
                }, 1000);
            } else {
                statusDiv.textContent = "API Error: " + (d.error || JSON.stringify(d.data));
            }
        } catch (e) {
            statusDiv.textContent = "Network Error.";
        }
        btn.disabled = false;
        btn.textContent = "Save Persona";
    });

    safeBind('ce-delete-btn', 'click', async () => {
        const slug = document.getElementById('ce-slug').value;
        if (!slug || !confirm("Permanently delete this persona from your Venice account?")) return;

        const btn = document.getElementById('ce-delete-btn');
        btn.disabled = true;
        btn.textContent = "Deleting...";

        try {
            const res = await fetch(`/delete_character/${slug}`, { method: 'DELETE' });
            const d = await res.json();
            if (d.success) {
                charEditorModal.style.display = 'none';
                fetchCharacters();
            } else {
                alert("Error deleting: " + (d.error || JSON.stringify(d)));
                btn.disabled = false;
                btn.textContent = "Delete";
            }
        } catch (e) {
            alert("Network Error.");
            btn.disabled = false;
            btn.textContent = "Delete";
        }
    });

    safeBind('close-characters-x', 'click', () => {
        if (charactersModal) charactersModal.style.display = 'none';
    });

    safeBind('char-search-btn', 'click', () => {
        fetchCharacters(charSearchInput.value);
    });

    if (charSearchInput) {
        charSearchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') fetchCharacters(charSearchInput.value);
        });
    }

    async function fetchCharacters(query = '') {
        const loading = document.getElementById('char-loading');
        const category = document.getElementById('char-category-filter')?.value || '';
        const modelId = document.getElementById('char-model-filter')?.value || '';

        if (loading) loading.style.display = 'block';
        if (charGalleryGrid) charGalleryGrid.innerHTML = '';

        try {
            let url = `/get_characters?search=${encodeURIComponent(query)}&category=${category}&modelId=${modelId}`;
            const res = await fetch(url);
            const data = await res.json();

            if (loading) loading.style.display = 'none';
            if (Array.isArray(data)) {
                if (!query && !category && !modelId) allCharacters = data;
                renderCharacterGallery(data);
            }
        } catch (e) {
            console.error("Character fetch failed:", e);
            if (loading) loading.style.display = 'none';
        }
    }

    function renderCharacterGallery(chars) {
        if (!charGalleryGrid) return;
        charGalleryGrid.innerHTML = '';

        chars.forEach(char => {
            const card = document.createElement('div');
            card.className = 'char-card';
            card.style.cssText = `
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 12px;
                padding: 15px;
                display: flex;
                flex-direction: column;
                gap: 10px;
                position: relative;
                transition: transform 0.2s, border-color 0.2s;
            `;

            card.onmouseenter = () => { card.style.transform = 'translateY(-3px)'; card.style.borderColor = 'var(--primary)'; };
            card.onmouseleave = () => { card.style.transform = 'translateY(0)'; card.style.borderColor = '#333'; };

            const topRow = document.createElement('div');
            topRow.style.display = 'flex';
            topRow.style.gap = '12px';
            topRow.style.alignItems = 'center';

            // Use the proxy for images to handle Bearer authentication
            const img = document.createElement('img');
            const rawImgUrl = char.photoUrl || 'https://venice.ai/placeholder-avatar.png';
            img.src = `/get_character_image?url=${encodeURIComponent(rawImgUrl)}`;
            img.style.cssText = 'width: 60px; height: 60px; border-radius: 50%; object-fit: cover; border: 2px solid #444;';

            const info = document.createElement('div');
            info.style.flex = '1';
            info.style.minWidth = '0';

            const ratingHtml = char.stats?.averageRating ? `<span style="color:#fbbf24; margin-left:5px;">â˜…${char.stats.averageRating.toFixed(1)}</span>` : '';

            info.innerHTML = `
                <div style="font-weight:bold; color:white; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-size:1.1em;">${char.name} ${ratingHtml}</div>
                <div style="font-size:0.75em; color:var(--text-dim);">@${char.slug} • by ${char.author || 'Anonymous'}</div>
                <div style="font-size:0.7em; color:var(--primary); margin-top:2px;">Opt: ${char.modelId || 'Universal'}</div>
            `;

            topRow.appendChild(img);
            topRow.appendChild(info);

            const desc = document.createElement('div');
            desc.style.cssText = 'font-size: 0.85em; color: #bbb; line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; height: 3.6em;';
            desc.textContent = char.description || "No description provided.";

            const statsRow = document.createElement('div');
            statsRow.style.cssText = 'display: flex; justify-content: space-between; align-items: center; margin-top: auto; border-top: 1px solid #2a2a2a; padding-top: 10px;';

            const count = char.stats?.imports || 0;
            statsRow.innerHTML = `<span style="font-size:0.75em; color:var(--text-dim);">${count.toLocaleString()} Imports</span>`;

            const btnGroup = document.createElement('div');
            btnGroup.style.display = 'flex';
            btnGroup.style.gap = '5px';

            const peekBtn = document.createElement('button');
            peekBtn.className = 'msg-btn';
            peekBtn.style.cssText = 'padding: 4px 8px; border-color: #555; font-size: 0.75em;';
            peekBtn.innerHTML = '👁️ Logic';
            peekBtn.title = "View the character's System Prompt / Instructions";
            peekBtn.onclick = (e) => {
                e.stopPropagation();
                openCharacterLogic(char.slug);
            };

            const useBtn = document.createElement('button');
            useBtn.className = 'msg-btn';
            useBtn.style.cssText = 'padding: 4px 12px; border-color: var(--primary); color: var(--primary); font-size: 0.75em; font-weight: bold;';
            useBtn.textContent = 'Use Persona';
            useBtn.onclick = (e) => {
                e.stopPropagation();
                selectCharacter(char);
            };

            btnGroup.appendChild(peekBtn);
            btnGroup.appendChild(useBtn);
            statsRow.appendChild(btnGroup);

            card.appendChild(topRow);
            card.appendChild(desc);
            card.appendChild(statsRow);
            charGalleryGrid.appendChild(card);
        });
    }

    async function openCharacterLogic(slug) {
        try {
            const res = await fetch(`/get_character_details/${slug}`);
            const char = await res.json();

            if (!char.instructions) {
                alert("Character logic is private or not provided.");
                return;
            }
            activeEditType = 'logic_view';

            // Inject Edit Button into the edit overlay
            const controls = document.getElementById('edit-controls');
            let editBtn = document.getElementById('char-quick-edit-btn');
            if (!editBtn) {
                editBtn = document.createElement('button');
                editBtn.id = 'char-quick-edit-btn';
                editBtn.className = 'action';
                editBtn.style.cssText = 'background:var(--purple); flex:0.5;';
                editBtn.textContent = 'Edit in Creator';
                controls.insertBefore(editBtn, controls.firstChild);
            }
            editBtn.style.display = 'block';
            editBtn.onclick = () => {
                editOverlay.style.display = 'none';
                openCharacterEditor(char);
            };

            editTextarea.value = char.instructions;
            editTextarea.readOnly = true;
            const saveBtn = document.getElementById('save-large-edit');
            if (saveBtn) saveBtn.style.display = 'none';

            editOverlay.style.display = 'block';

            // Handle closing cleanup
            const originalClose = document.getElementById('cancel-large-edit').onclick;
            document.getElementById('cancel-large-edit').onclick = () => {
                editTextarea.readOnly = false;
                if (saveBtn) saveBtn.style.display = 'block';
                const eb = document.getElementById('char-quick-edit-btn');
                if (eb) eb.style.display = 'none';
                editOverlay.style.display = 'none';
                document.getElementById('cancel-large-edit').onclick = originalClose;
            };
        } catch (e) {
            alert("Failed to fetch detailed character logic.");
        }
    }

    async function selectCharacter(char) {
        let confirmMsg = `Switch to persona: ${char.name}?`;
        if (char.modelId) {
            confirmMsg += `\n\nNote: This will also set your chat model to ${char.modelId} for best performance.`;
        }

        if (!confirm(confirmMsg)) return;

        try {
            const res = await fetch('/update_character', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ 
                    slug: char.slug,
                    modelId: char.modelId 
                })
            });
            if (res.ok) {
                if (charactersModal) charactersModal.style.display = 'none';
                updateActiveCharacterUI(char);
                if (confirm("Character selected. Would you like to start a fresh chat with this persona?")) {
                    await fetch('/new_chat', {method:'POST'});
                }
                await loadHistory();
            }
        } catch (e) {
            alert("Failed to select character.");
        }
    }

    function updateActiveCharacterUI(char) {
        const info = document.getElementById('active-character-info');
        const img = document.getElementById('active-char-img');
        const name = document.getElementById('active-char-name');

        if (!char) {
            if (info) info.style.display = 'none';
            return;
        }

        if (info && img && name) {
            info.style.display = 'block';
            const rawImgUrl = char.photoUrl || 'https://venice.ai/placeholder-avatar.png';
            img.src = `/get_character_image?url=${encodeURIComponent(rawImgUrl)}`;
            name.textContent = char.name;
        }
    }

    safeBind('clear-character-btn', 'click', async () => {
        if (!confirm("Remove character persona? The chat will return to the default system prompt.")) return;
        try {
            await fetch('/update_character', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ slug: null })
            });
            updateActiveCharacterUI(null);
            await loadHistory();
        } catch (e) {}
    });

    async function loadHistory() {
        try {
            const sbRes = await fetch('/sidebar_data');
            const sbData = await sbRes.json();
            isAuditMode = sbData.active_chat && sbData.active_chat.endsWith('.audit.json');
            
            const auditTopBar = document.getElementById('audit-top-bar');
            if (auditTopBar) {
                auditTopBar.style.display = isAuditMode ? 'flex' : 'none';
                if (chatbox) chatbox.style.paddingTop = isAuditMode ? '115px' : '70px';
            }
            if (!isAuditMode) {
                const drawer = document.getElementById('audit-drawer');
                if (drawer) drawer.style.display = 'none';
            }

            const res = await fetch('/get_history');
            const data = await res.json();
            
            if (data.is_arena) {
                window.arenaData = data;
                window.isArenaMode = true;
                
                document.getElementById('arena-toolbar').style.display = 'flex';
                document.getElementById('chatbox').style.paddingTop = '115px';
                
                const compareContainer = document.getElementById('compare-mode-container');
                if (compareContainer) compareContainer.style.display = 'none';
                
                // VALIDATION: Ensure activeArenaTab is valid for the current arenaData models
                const isValidTab = window.activeArenaTab === 'evaluator' || 
                                 (window.arenaData.models && window.arenaData.models.some(m => m.id === window.activeArenaTab));
                
                if (!window.activeArenaTab || !isValidTab) {
                    window.activeArenaTab = data.models[0].id;
                }
                
                // Populate Evaluator Model Select if empty
                const evalSelect = document.getElementById('arena-eval-model-select');
                if (evalSelect && evalSelect.options.length === 0 && availableModels) {
                    for (const group in availableModels) {
                        const optgroup = document.createElement('optgroup');
                        optgroup.label = group;
                        availableModels[group].forEach(m => {
                            const opt = document.createElement('option');
                            opt.value = m.id;
                            opt.textContent = m.name;
                            if (m.id === 'venice-uncensored') opt.selected = true;
                            optgroup.appendChild(opt);
                        });
                        evalSelect.appendChild(optgroup);
                    }
                }

                renderArenaChat();
                return;
            } else {
                window.isArenaMode = false;
                const toolbar = document.getElementById('arena-toolbar');
                if (toolbar) toolbar.style.display = 'none';
                const aRow = document.getElementById('arena-controls-row');
                if (aRow) aRow.style.display = 'none';
                const eRow = document.getElementById('evaluator-controls-row');
                if (eRow) eRow.style.display = 'none';
                document.getElementById('chatbox').style.paddingTop = '70px';
            }

            if (isAuditMode && data.audit_context) {
                auditContextSelection = data.audit_context;
                updateAuditTopBar();
                                fetchParentContext();
            }

            // [INJECTION] Check for Pipeline
            const openBtn = document.getElementById('open-pipeline-panel-btn');
            const chatHeader = document.getElementById('chat-header');
            if (data.chat_type === 'pipeline') {
                if (chatHeader) {
                    chatHeader.style.display = 'flex';
                    const nameSpan = document.getElementById('header-chat-name');
                    if (nameSpan && sbData.active_chat) {
                        nameSpan.textContent = sbData.active_chat.replace('.json', '').replace(/_/g, ' ');
                    }
                }
                if (openBtn) openBtn.style.display = 'block';
                if (chatbox) chatbox.style.paddingTop = '60px'; // Adjust for header
                
                // Sync Content to Modal
                const editor = document.getElementById('blueprint-editor');
                if (editor) {
                    editor.value = data.blueprint || '';
                    previousBlueprint = data.blueprint || '';
                }
                
                currentPipelinePhase = data.pipeline_phase || 'architect';
                
                // Update Badge (Header)
                const phaseBadge = document.getElementById('pipeline-phase-badge');
                const trackIndicator = document.getElementById('pipeline-track-indicator');
                if (phaseBadge) {
                    phaseBadge.textContent = currentPipelinePhase.toUpperCase();
                    phaseBadge.style.background = currentPipelinePhase === 'scribe' ? 'var(--purple)' : 'var(--accent)';
                }
                if (trackIndicator) {
                    trackIndicator.textContent = currentPipelinePhase === 'scribe' ? '(Actual Story Prose)' : '(Brainstorming & Blueprinting)';
                }

                // Update Badge (Modal)
                const badge = document.getElementById('pipeline-status-badge');
                if (badge) {
                    badge.textContent = currentPipelinePhase.toUpperCase();
                    badge.style.background = currentPipelinePhase === 'scribe' ? 'var(--purple)' : 'var(--accent)';
                }
                
                // Update Mode Label
                const modeLabel = document.getElementById('pipeline-mode-label');
                if (modeLabel) modeLabel.textContent = currentPipelinePhase.charAt(0).toUpperCase() + currentPipelinePhase.slice(1);

                // Update Lock Button
                const lockBtn = document.getElementById('lock-scribe-btn');
                if (lockBtn) {
                    if (currentPipelinePhase === 'scribe') {
                        lockBtn.textContent = '← Back to Architect';
                        if (editor) editor.readOnly = true;
                    } else {
                        lockBtn.textContent = 'Lock & Scribe';
                        if (editor) editor.readOnly = false;
                    }
                }
            } else {
                if (chatHeader) chatHeader.style.display = 'none';
                if (openBtn) openBtn.style.display = 'none';
                if (!window.isArenaMode && !isAuditMode) {
                    if (chatbox) chatbox.style.paddingTop = '70px';
                }
                const docModal = document.getElementById('pipeline-doc-modal');
                if (docModal) docModal.style.display = 'none';
                document.body.style.overflow = '';
            }

            chatHistory = data.history || [];
            summaries = data.summaries || [];
            currentVisualMemory = data.visual_memory || "";
            if (vmTextarea) vmTextarea.value = currentVisualMemory;

            currentSelfMemory = data.self_memory || "";
            memoryLogs = data.memory_logs || [];
            if (document.getElementById('memory-modal')?.style.display === 'block') {
                renderMemoryManagerUI();
            }

            // Character UI handling
            if (data.character_slug) {
                let char = allCharacters.find(c => c.slug === data.character_slug);
                if (char) {
                    updateActiveCharacterUI(char);
                } else {
                    updateActiveCharacterUI({ name: data.character_slug, slug: data.character_slug });
                }

                // SIMULATED GREETING: If chat is brand new (only system prompt)
                if (chatHistory.length === 1 && chatHistory[0].role === 'system') {
                    // Slight delay to ensure UI is updated
                    setTimeout(() => triggerGen("Initialize our conversation with a characteristic greeting in character."), 500);
                }
            } else {
                updateActiveCharacterUI(null);
            }

            const compareContainer = document.getElementById('compare-mode-container');
            if (compareContainer) {
                if (data.has_backup) {
                    compareContainer.style.display = 'block';
                } else {
                    compareContainer.style.display = 'none';
                    const cb = document.getElementById('compare-mode-cb');
                    if(cb) cb.checked = false;
                }
            }

            renderChat();
        } catch (e) { console.error("History error", e); }
    }

    let lastWfmText = "";
    let lastWfmUsage = null;

    function getModelPricing(modelId) {
        if (!availableModels) return null;
        for (const group in availableModels) {
            const found = availableModels[group].find(m => m.id === modelId);
            if (found && found.pricing) {
                const parts = found.pricing.split('(');
                const basePrices = parts[0].replace(/[\$\s]/g, '').split('/');
                const inputRate = parseFloat(basePrices[0]) || 0;
                const outputRate = parseFloat(basePrices[1]) || 0;
                
                let cacheRate = inputRate * 0.5; // Default 50% for Venice
                if (parts.length > 1) {
                    const cachePart = parts[1].match(/\$?([0-9.]+)/);
                    if (cachePart) cacheRate = parseFloat(cachePart[1]);
                }
                return { input: inputRate, output: outputRate, cache: cacheRate };
            }
        }
        return null;
    }

    function renderChat(forceScroll = false) {
        if (!chatbox) return;
        
        // Save current scroll state
        const isAtBottom = chatbox.scrollHeight - chatbox.scrollTop <= chatbox.clientHeight + 100;
        
        chatbox.innerHTML = '';
        const fragment = document.createDocumentFragment();

        runningInput = 0;
        runningOutput = 0;
        let textMsgCounter = 0;

        // --- SUMMARIZATION UI LOGIC ---
        const liveSumEnabled = document.getElementById('live-sum-enabled')?.checked || false;
        const keepTurns = parseInt(document.getElementById('sum-keep')?.value) || 12;
        
        let lastSumIndex = -1;
        let liveThresholdIdx = -1;

        if (liveSumEnabled) {
            const batchSize = parseInt(document.getElementById('sum-batch')?.value) || 4;
            let validIndices = [];
            for (let i = 0; i < chatHistory.length; i++) {
                if (chatHistory[i].role !== 'system' && !(typeof chatHistory[i].content === 'string' && chatHistory[i].content.startsWith('__IMG_JSON__'))) {
                    validIndices.push(i);
                }
            }
            let V = validIndices.length;
            let summarizableCount = V - keepTurns;
            if (summarizableCount > 0) {
                let summarizedCount = Math.floor(summarizableCount / batchSize) * batchSize;
                if (summarizedCount > 0) {
                    liveThresholdIdx = validIndices[summarizedCount - 1];
                }
            }
        } else if (summaries && summaries.length > 0) {
            lastSumIndex = summaries[summaries.length - 1].end_index;
        }

        chatHistory.forEach((msg, idx) => {
            if (msg.role === 'system') return;

            const isImage = typeof msg.content === 'string' && msg.content.startsWith('__IMG_JSON__');
            if (!isImage) textMsgCounter++;

            if (msg.usage) {
                runningInput += (msg.usage.prompt_tokens || 0);
                runningOutput += (msg.usage.completion_tokens || 0);
            }

            const div = document.createElement('div');
            div.className = `message ${msg.role}`;
            div.id = `msg-${idx}`;

            if (msg.timestamp) {
                const ts = document.createElement('div');
                ts.className = 'msg-timestamp';
                ts.textContent = new Date(msg.timestamp).toLocaleTimeString([], { hour12: true, hour: 'numeric', minute: '2-digit', second: '2-digit' });
                div.appendChild(ts);
            }

            // Determine if message is visually "archived/summarized"
            const isSummed = liveSumEnabled ? (idx <= liveThresholdIdx) : (idx <= lastSumIndex);
            if (isSummed) div.classList.add('summarized-msg');

            if (isImage) {
                try {
                    const imgData = JSON.parse(msg.content.replace('__IMG_JSON__', ''));
                    const imgEl = document.createElement('img');
                    imgEl.src = imgData.url;
                    imgEl.className = 'img-display';

                    const promptBox = document.createElement('div');
                    promptBox.className = 'prompt-box';
                    promptBox.innerHTML = `<div><strong>Prompt:</strong> ${imgData.prompt}</div>`;

                    const delGenBtn = document.createElement('button');
                    delGenBtn.className = 'msg-btn';
                    delGenBtn.style.cssText = 'background:var(--danger); color:white; border:none; margin-top:10px; padding:6px 12px;';
                    delGenBtn.textContent = '🗑️ Delete Image';
                    delGenBtn.onclick = async (e) => {
                        e.stopPropagation();
                        if (confirm("Permanently delete this generated image?")) {
                            chatHistory.splice(idx, 1);
                            await saveHistory();
                            renderChat();
                        }
                    };
                    promptBox.appendChild(delGenBtn);

                    imgEl.onclick = () => {
                        promptBox.style.display = promptBox.style.display === 'none' ? 'block' : 'none';
                    };

                    div.appendChild(imgEl);
                    div.appendChild(promptBox);
                } catch(e) {
                    div.textContent = "Error loading image.";
                }
            } else {
                try {
                    // Reasoning Block
                    if (msg.reasoning) {
                        const rBlock = document.createElement('div');
                        rBlock.className = 'reasoning-block';
                        rBlock.innerHTML = `
                            <div class="reasoning-header">
                                <span><i style="margin-right:5px;">💭</i> Thought Process</span>
                                <span class="toggle-icon">▼</span>
                            </div>
                            <div class="reasoning-content" style="display:none;">${msg.reasoning}</div>
                        `;
                        rBlock.querySelector('.reasoning-header').onclick = () => {
                            const content = rBlock.querySelector('.reasoning-content');
                            const icon = rBlock.querySelector('.toggle-icon');
                            const isHidden = content.style.display === 'none';
                            content.style.display = isHidden ? 'block' : 'none';
                            icon.textContent = isHidden ? '▲' : '▼';
                        };
                        div.appendChild(rBlock);
                    }
                    
                    // Live Summary Block
                    if (msg.live_summary) {
                        const sBlock = document.createElement('div');
                        sBlock.className = 'summary-block';
                        sBlock.innerHTML = `
                            <div class="summary-header">
                                <span><i style="margin-right:5px;">📝</i> Live Inline Summary</span>
                                <span class="toggle-icon-sum">▼</span>
                            </div>
                            <div class="summary-content" style="display:none;">${msg.live_summary}</div>
                        `;
                        sBlock.querySelector('.summary-header').onclick = () => {
                            const content = sBlock.querySelector('.summary-content');
                            const icon = sBlock.querySelector('.toggle-icon-sum');
                            const isHidden = content.style.display === 'none';
                            content.style.display = isHidden ? 'block' : 'none';
                            icon.textContent = isHidden ? '▲' : '▼';
                        };
                        div.appendChild(sBlock);
                    }

                    const contentDiv = document.createElement('div');
                    contentDiv.style.cssText = "white-space:pre-wrap; font-family:inherit;";
                    contentDiv.className = 'msg-content';

                    // Tool Calls Block (Metadata view)
                    if (msg.blueprint_tool_calls) {
                        const tcBlock = document.createElement('div');
                        tcBlock.className = 'tool-calls-block';
                        tcBlock.style.cssText = 'margin-bottom:12px; border:1px solid #2d4a2d; border-radius:6px; overflow:hidden; font-size:0.85em;';
                        tcBlock.innerHTML = `
                            <div class="tool-calls-header" style="display:flex; justify-content:space-between; align-items:center; padding:6px 12px; background:#1a2e1a; cursor:pointer; user-select:none;">
                                <span style="color:#4ade80;"><i style="margin-right:5px;">🛠️</i> Blueprint Actions</span>
                                <span class="toggle-icon-tc">▼</span>
                            </div>
                            <div class="tool-calls-content" style="display:none; padding:12px; background:#0a0a0a; white-space:pre-wrap; font-family:monospace; color:#a3e635; border-top:1px solid #2d4a2d;">${msg.blueprint_tool_calls}</div>
                        `;
                        tcBlock.querySelector('.tool-calls-header').onclick = () => {
                            const c = tcBlock.querySelector('.tool-calls-content');
                            const ic = tcBlock.querySelector('.toggle-icon-tc');
                            const isH = c.style.display === 'none';
                            c.style.display = isH ? 'block' : 'none';
                            ic.textContent = isH ? '▲' : '▼';
                        };
                        div.appendChild(tcBlock);
                    }

                    if (diffModeMsgs.has(idx) && msg.original_content) {
                        const origText = Array.isArray(msg.original_content) ? msg.original_content.find(p=>p.type==='text')?.text || "" : msg.original_content;
                        const curText = Array.isArray(msg.content) ? msg.content.find(p=>p.type==='text')?.text || "" : msg.content;
                        contentDiv.innerHTML = renderDiff(origText, curText);
                    } else {
                        let textContent = Array.isArray(msg.content) ? (msg.content.find(p => p.type === 'text')?.text || "") : msg.content;
                        
                        // Parse markdown first so we don't break code blocks or standard formatting
                        let parsedContent = typeof marked !== 'undefined' ? marked.parse(textContent || "") : textContent;
                        
                        // Detect and transform the Audit tool tags into Comparison Cards
                        const auditRegex = /\[UPDATE_SUMMARY index=["&quot;]*(\d+)["&quot;]*\]([\s\S]*?)\[\/UPDATE_SUMMARY\]/gi;
                        parsedContent = parsedContent.replace(auditRegex, (match, idx, newText) => {
                            const cleanNew = newText.trim();
                            let oldTextHTML = `<div style="color:#888; font-style:italic;">(Original summary data not loaded. Open Evidence Drawer to sync.)</div>`;
                            
                            const bIdx = parseInt(idx);
                            if (parentSummariesCache && parentSummariesCache[bIdx]) {
                                const oldText = parentSummariesCache[bIdx].content.replace(/<think>.*?<\/think>/gs, '').trim();
                                oldTextHTML = `<div><strong>Original Summary:</strong><br><div class="audit-comparison-panel">${oldText}</div></div>`;
                            }
                            
                            return `<div class="audit-tool-card">
                                <div class="audit-tool-header">ðŸ› ï¸ Proposed Summary Update (Batch #${bIdx + 1})</div>
                                ${oldTextHTML}
                                <div><strong>Proposed Fix:</strong><br><div class="audit-comparison-panel" style="border-left-color:#10b981;">${cleanNew}</div></div>
                                <div style="display:flex; gap:10px; margin-top:12px;">
                                    <button class="msg-btn" style="flex:1; background:#333; border:1px solid #555;" onclick="document.getElementById('audit-top-bar').click();">View Sources</button>
                                    <button class="msg-btn apply-audit-btn" data-index="${idx}" data-text="${encodeURIComponent(cleanNew)}" style="flex:2; background:#2563eb; color:white; border:none; font-weight:bold;">Approve & Apply to Original Chat</button>
                                </div>
                            </div>`;
                        });

                        contentDiv.innerHTML = parsedContent;

                        if (Array.isArray(msg.content)) {
                            msg.content.forEach((part, partIdx) => {
                                if (part.type === 'image_url') {
                                    const wrapper = document.createElement('div');
                                    wrapper.className = 'input-img-wrapper';

                                    const imgEl = document.createElement('img');
                                    imgEl.src = part.image_url.url;
                                    imgEl.className = 'input-img';

                                    const overlay = document.createElement('div');
                                    overlay.className = 'input-img-overlay';

                                    const resToggle = document.createElement('button');
                                    resToggle.className = 'img-action-btn';
                                    resToggle.textContent = (part.image_url.detail === 'high') ? 'HD' : 'SD';
                                    resToggle.title = 'Toggle Resolution (High/Low)';
                                    resToggle.onclick = async () => {
                                        part.image_url.detail = (part.image_url.detail === 'high') ? 'low' : 'high';
                                        await saveHistory();
                                        renderChat();
                                    };

                                    const delBtn = document.createElement('button');
                                    delBtn.className = 'img-action-btn delete';
                                    delBtn.textContent = '🗑️';
                                    delBtn.title = 'Delete Image';
                                    delBtn.onclick = async () => {
                                        if (confirm("Permanently delete this input image from the message?")) {
                                            msg.content.splice(partIdx, 1);
                                            if (msg.content.length === 1 && msg.content[0].type === 'text') {
                                                msg.content = msg.content[0].text;
                                            }
                                            await saveHistory();
                                            renderChat();
                                        }
                                    };

                                    overlay.appendChild(resToggle);
                                    overlay.appendChild(delBtn);
                                    wrapper.appendChild(imgEl);
                                    wrapper.appendChild(overlay);
                                    contentDiv.appendChild(wrapper);
                                }
                            });
                        }
                    }
                    div.appendChild(contentDiv);
                } catch(e) {
                    console.error("Markdown Error:", e);
                    div.textContent = msg.content;
                }
            }

            const optionsBtn = document.createElement('div');
            optionsBtn.className = 'options-btn';
            optionsBtn.innerHTML = '&#8942;'; // Vertical Ellipsis

            const toolsPanel = document.createElement('div');
            toolsPanel.className = 'tools-panel';

            const btnRow = document.createElement('div');
            btnRow.className = 'tools-row';

            const editBtn = document.createElement('button'); editBtn.className = 'msg-btn'; editBtn.textContent = 'Edit';
            editBtn.onclick = () => openLargeEditor(idx, 'message');

            const regenBtn = document.createElement('button'); regenBtn.className = 'msg-btn'; regenBtn.textContent = msg.role === 'user' ? 'Resend' : 'Regen';
            regenBtn.onclick = () => handleRegen(idx, msg.role);

            const branchBtn = document.createElement('button'); branchBtn.className = 'msg-btn'; branchBtn.textContent = 'Branch Here';
            branchBtn.onclick = () => handleBranch(idx);

            const branchArenaBtn = document.createElement('button'); 
            branchArenaBtn.className = 'msg-btn'; 
            branchArenaBtn.style.borderColor = 'var(--purple)';
            branchArenaBtn.textContent = 'Branch to Arena';
            branchArenaBtn.onclick = () => handleBranchToArena(idx);

            const refineBtn = document.createElement('button'); refineBtn.className = 'msg-btn'; refineBtn.textContent = 'AI Refine';
            refineBtn.onclick = () => handleAIRefine(idx);

            const findReplaceBtn = document.createElement('button'); 
            findReplaceBtn.className = 'msg-btn'; 
            findReplaceBtn.textContent = 'Find/Replace';
            findReplaceBtn.onclick = () => handleFindReplace(idx);

            const delBtn = document.createElement('button'); 
            delBtn.className = 'msg-btn delete-msg-btn'; 
            delBtn.textContent = 'Delete';
            delBtn.onclick = () => handleDeleteMessage(idx);

            btnRow.append(editBtn, regenBtn);
            if (msg.role === 'assistant' && !isImage) {
                const regenModelBtn = document.createElement('button'); 
                regenModelBtn.className = 'msg-btn'; 
                regenModelBtn.style.borderColor = 'var(--purple)';
                regenModelBtn.style.color = 'var(--purple)';
                regenModelBtn.textContent = 'Regen w/ Model';
                regenModelBtn.onclick = () => handleRegenWithModel(idx);
                btnRow.append(regenModelBtn);
            }

            if (window.ttsSettings && window.ttsSettings.enabled && !isImage) {
                const ttsBtn = document.createElement('button');
                ttsBtn.className = 'msg-btn tts-btn';
                ttsBtn.dataset.index = idx;
                ttsBtn.style.borderColor = 'var(--accent)';
                ttsBtn.style.color = 'var(--accent)';
                ttsBtn.textContent = '▶ Narrate';
                ttsBtn.onclick = () => handleTTS(idx);
                btnRow.append(ttsBtn);
            }

            btnRow.append(branchBtn, branchArenaBtn, refineBtn, findReplaceBtn, delBtn);
            toolsPanel.appendChild(btnRow);

            // After appending everything to toolsPanel, update state if this was playing
            setTimeout(() => {
                updateTTSButtonStates();
                if (currentTTS.msgIndex === idx) renderTTSControls(idx);
            }, 0);

            if (msg.original_content) {
                const diffRow = document.createElement('div');
                diffRow.className = 'tools-row';
                diffRow.style.marginTop = '5px';

                const toggleBtn = document.createElement('button');
                toggleBtn.className = 'msg-btn';
                toggleBtn.style.background = 'var(--accent)';
                toggleBtn.style.borderColor = 'var(--accent)';
                toggleBtn.textContent = diffModeMsgs.has(idx) ? 'Hide Changes' : 'Show Changes';
                toggleBtn.onclick = () => {
                    if (diffModeMsgs.has(idx)) diffModeMsgs.delete(idx);
                    else diffModeMsgs.add(idx);
                    renderChat();
                };

                const undoBtn = document.createElement('button');
                undoBtn.className = 'msg-btn';
                undoBtn.style.background = 'var(--danger)';
                undoBtn.style.borderColor = 'var(--danger)';
                undoBtn.textContent = 'Undo Refine';
                undoBtn.onclick = () => handleUndoRefine(idx);

                diffRow.append(toggleBtn, undoBtn);

                if (msg.refine_logic) {
                    const logicBtn = document.createElement('button');
                    logicBtn.className = 'msg-btn';
                    logicBtn.style.background = 'var(--purple)';
                    logicBtn.style.borderColor = 'var(--purple)';
                    logicBtn.textContent = 'View Refinement Logic';
                    logicBtn.onclick = () => {
                        const modal = document.getElementById('logic-view-modal');
                        const content = document.getElementById('logic-view-content');
                        if (modal && content) {
                            content.textContent = msg.refine_logic;
                            modal.style.display = 'block';
                        }
                    };
                    diffRow.append(logicBtn);
                }

                toolsPanel.appendChild(diffRow);
            }

            const stats = document.createElement('div');
            stats.className = 'stats-block';

            let statHtml = "";
            if (isImage) {
                statHtml = `<span style="color:var(--accent);">Generated Image</span>`;
            } else {
                if (msg.comparison_choice) {
                    const label = msg.comparison_choice === 'A' ? 'Current' : 'Original';
                    const color = msg.comparison_choice === 'A' ? 'var(--accent)' : 'var(--purple)';
                    statHtml += `<span class="compare-badge" style="border-left: 3px solid ${color};">Resolved: ${label}</span><br>`;
                }
                statHtml += `Msg #${textMsgCounter}`;
                if (isSummed) statHtml += ` <span class="sum-badge">SUMMARIZED</span>`;

                let textContentForCount = Array.isArray(msg.content) ? (msg.content.find(p => p.type === 'text')?.text || "") : msg.content;
                statHtml += ` <span style="color:var(--text-dim); margin-left:8px;">[${textContentForCount.length} chars]</span>`;

                if (msg.model) statHtml += `<br>Model: <span style="color:var(--primary);">${msg.model}</span>`;

                if (msg.usage) {
                    const u = msg.usage;
                    if (msg.role === 'assistant') {
                        let reasoningTokens = 0;
                        if (u.completion_tokens_details && u.completion_tokens_details.reasoning_tokens) {
                            reasoningTokens = u.completion_tokens_details.reasoning_tokens;
                        }

                        statHtml += `<br>Usage: In:${u.prompt_tokens} Out:${u.completion_tokens}`;

                        if (reasoningTokens > 0) {
                            let answerTokens = u.completion_tokens - reasoningTokens;
                            statHtml += ` <span style="font-size:0.85em; color:var(--purple);">(Think: ${reasoningTokens} | Answer: ${answerTokens})</span>`;
                        }

                        if (u.breakdown) {
                            statHtml += `<br><span style="font-size:0.85em; color:var(--text-dim);">â”œ Sum: ${u.breakdown.summary || 0}</span>`;
                            statHtml += `<br><span style="font-size:0.85em; color:var(--text-dim);">â”œ Raw: ${u.breakdown.raw || 0}</span>`;
                            statHtml += `<br><span style="font-size:0.85em; color:var(--text-dim);">â”” Sys: ${u.breakdown.system || 0}</span>`;
                        }

                        let cachedTokens = 0;
                        if (u.prompt_tokens_details && u.prompt_tokens_details.cached_tokens) {
                            cachedTokens = u.prompt_tokens_details.cached_tokens;
                        }
                        if (cachedTokens > 0) {
                            statHtml += `<br><span style="font-size:0.85em; color:#10b981;">Cached: ${cachedTokens} (${Math.round((cachedTokens/u.prompt_tokens)*100)}%)</span>`;
                        }

                        // COST CALCULATION
                        const pricing = getModelPricing(msg.model);
                        if (pricing) {
                            const inputCost = ((u.prompt_tokens - cachedTokens) / 1000000) * pricing.input;
                            const cacheCost = (cachedTokens / 1000000) * pricing.cache;
                            const outputCost = (u.completion_tokens / 1000000) * pricing.output;
                            const totalCost = inputCost + cacheCost + outputCost;

                            statHtml += `<br><span class="cost-wrapper" style="cursor:pointer; color:var(--text-dim);" onclick="const val = this.querySelector('.cost-value'); const placeholder = this.querySelector('.cost-placeholder'); if(val.style.display === 'none') { val.style.display = 'inline'; placeholder.style.display = 'none'; this.style.color = 'inherit'; } else { const details = this.nextElementSibling; details.style.display = (details.style.display === 'none' ? 'block' : 'none'); }">Cost: <span class="cost-placeholder" style="text-decoration:underline dotted;">[Show]</span><span class="cost-value" style="display:none; text-decoration:underline dashed; color:var(--accent);">${totalCost.toFixed(5)}</span></span>`;
                            statHtml += `<div class="cost-details" style="display:none; font-size:0.85em; color:var(--text-dim); margin-top:4px; padding-left:10px; border-left:1px solid #444;">`;
                            statHtml += `â”œ Input: ${inputCost.toFixed(5)}<br>`;
                            if (cachedTokens > 0) statHtml += `â”œ Cache: ${cacheCost.toFixed(5)}<br>`;
                            statHtml += `â”” Output: ${outputCost.toFixed(5)}`;
                            statHtml += `</div>`;
                        }

                    } else if (msg.role === 'user') {
                        statHtml += `<br>WFM Usage: In:${u.prompt_tokens} Out:${u.completion_tokens}`;
                    }
                }
                statHtml += `<br>Running: <span class="stat-highlight">In:${runningInput} Out:${runningOutput}</span>`;
            }
            stats.innerHTML = statHtml;

            if (isSummed && !isImage) {
                const myBatch = summaries.find(s => idx >= s.start_index && idx <= s.end_index);
                if (myBatch && !msg.live_summary) {
                    const viewSumBtn = document.createElement('button');
                    viewSumBtn.className = 'msg-btn';
                    viewSumBtn.style.marginTop = '5px';
                    viewSumBtn.style.width = '100%';
                    viewSumBtn.textContent = 'View Batch Summary';
                    viewSumBtn.onclick = () => viewSummary(myBatch);
                    stats.appendChild(viewSumBtn);
                }
            }

            toolsPanel.appendChild(stats);
            optionsBtn.onclick = () => { toolsPanel.style.display = (toolsPanel.style.display === 'block') ? 'none' : 'block'; };
            div.appendChild(optionsBtn);
            div.appendChild(toolsPanel);
            fragment.appendChild(div);
        });
        
        chatbox.appendChild(fragment);

        if (forceScroll || (isAtBottom && !isGenerating)) {
            chatbox.scrollTop = chatbox.scrollHeight;
        }
    }

    function formatSummaryText(text) {
        if (text.includes('<think>')) {
            const parts = text.split(/<think>|<\/think>/);
            if (parts.length >= 3) {
                const thinking = parts[1].trim();
                const content = parts.slice(2).join('').trim();
                return `
                    <div class="reasoning-block">
                        <div class="reasoning-header">
                            <span><i style="margin-right:5px;">💭</i> Summary Thought Process</span>
                            <span class="toggle-icon">▼</span>
                        </div>
                        <div class="reasoning-content" style="display:none;">${thinking}</div>
                    </div>
                    <div style="white-space:pre-wrap;">${content}</div>
                `;
            }
        }
        return `<div style="white-space:pre-wrap;">${text}</div>`;
    }

    function viewSummary(batch) {
        if (!batch) return;
        const managerControls = document.getElementById('summary-manager-controls');
        if (managerControls) managerControls.style.display = 'none';

        const container = document.getElementById('summary-content');
        if (!container) return;
        container.innerHTML = '';

        let textMsgsUpToStart = chatHistory.slice(0, batch.start_index).filter(m => 
            m.role !== 'system' && !(typeof m.content === 'string' && m.content.startsWith('__IMG_JSON__'))
        ).length;
        let textMsgsInBatch = chatHistory.slice(batch.start_index, batch.end_index + 1).filter(m => 
            m.role !== 'system' && !(typeof m.content === 'string' && m.content.startsWith('__IMG_JSON__'))
        ).length;
        let startNum = textMsgsUpToStart + 1;
        let endNum = textMsgsUpToStart + textMsgsInBatch;
        let usage = batch.usage ? `In: ${batch.usage.prompt_tokens || 0}, Out: ${batch.usage.completion_tokens || 0}` : "N/A";

        let info = `[Batch: Messages #${startNum} - #${endNum}]\n[Token Usage: ${usage}]\n\n`;
        if (batch.is_consolidated) {
            info = `[CONSOLIDATED BATCH: Messages #${startNum} - #${endNum}]\n[Token Usage: ${usage}]\n\n`;
        }

        const infoDiv = document.createElement('div');
        infoDiv.style.color = 'var(--text-dim)';
        infoDiv.style.fontSize = '0.9em';
        infoDiv.style.marginBottom = '10px';
        infoDiv.textContent = info;
        container.appendChild(infoDiv);

        const contentDiv = document.createElement('div');
        contentDiv.innerHTML = formatSummaryText(batch.content);

        // Re-bind accordion logic for the dynamically created elements
        const header = contentDiv.querySelector('.reasoning-header');
        if (header) {
            header.onclick = () => {
                const c = contentDiv.querySelector('.reasoning-content');
                const i = contentDiv.querySelector('.toggle-icon');
                const isH = c.style.display === 'none';
                c.style.display = isH ? 'block' : 'none';
                i.textContent = isH ? '▲' : '▼';
            };
        }

        container.appendChild(contentDiv);

        const controls = document.getElementById('summary-controls');
        if (controls) {
            controls.style.display = 'flex';
            const regenBtn = document.getElementById('regen-batch-btn');
            if (regenBtn) {
                regenBtn.onclick = async () => {
                    if(batch.is_consolidated) {
                        alert("Cannot regenerate a consolidated summary. Undo the consolidation first.");
                        return;
                    }
                    if(!confirm("Regenerate this specific batch?")) return;
                    const ogText = regenBtn.textContent;
                    regenBtn.textContent = "Regenerating..."; regenBtn.disabled = true;
                    try {
                        const batchIndex = summaries.indexOf(batch);
                        const res = await fetch('/force_summarize', {
                            method: 'POST', headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({mode: 'batch', batch_index: batchIndex})
                        });
                        const d = await res.json();
                        if(d.success) {
                            await loadHistory();
                            const updatedBatch = summaries[batchIndex];
                            viewSummary(updatedBatch);
                        } else {
                            alert("Error: " + (d.error || "Unknown server error"));
                        }
                    } catch(e) { 
                        console.error(e);
                        alert("Failed to connect to server.");
                    }
                    regenBtn.textContent = ogText; regenBtn.disabled = false;
                };
            }
        }

        const sumModal = document.getElementById('summary-view-modal');
        if (sumModal) sumModal.style.display = 'block';
    }

    function renderMemoryManager() {
        if (!summaries || summaries.length === 0) { alert("No summaries exist yet."); return; }
        const container = document.getElementById('summary-content');
        if (!container) return;
        container.innerHTML = '';
        const controls = document.getElementById('summary-controls');
        if (controls) controls.style.display = 'none';

        const mgrControls = document.getElementById('summary-manager-controls');
        if (mgrControls) mgrControls.style.display = 'flex';

        const rangeStart = document.getElementById('range-start');
        const rangeEnd = document.getElementById('range-end');
        if(rangeStart && rangeEnd) {
            rangeStart.max = summaries.length;
            rangeEnd.max = summaries.length;
            rangeStart.value = '';
            rangeEnd.value = '';
        }

        summaries.forEach((batch, i) => {
            const isConsolidated = batch.is_consolidated;

            let textMsgsUpToStart = chatHistory.slice(0, batch.start_index).filter(m => 
                m.role !== 'system' && !(typeof m.content === 'string' && m.content.startsWith('__IMG_JSON__'))
            ).length;
            let textMsgsInBatch = chatHistory.slice(batch.start_index, batch.end_index + 1).filter(m => 
                m.role !== 'system' && !(typeof m.content === 'string' && m.content.startsWith('__IMG_JSON__'))
            ).length;
            let startNum = textMsgsUpToStart + 1;
            let endNum = textMsgsUpToStart + textMsgsInBatch;

            const card = document.createElement('div');
            card.style.background = '#1a1a1a';
            card.style.border = '1px solid #333';
            card.style.borderRadius = '8px';
            card.style.padding = '15px';
            card.style.marginBottom = '10px';
            card.style.position = 'relative';

            const header = document.createElement('div');
            header.style.display = 'flex';
            header.style.justifyContent = 'space-between';
            header.style.alignItems = 'center';
            header.style.marginBottom = '10px';
            header.style.borderBottom = '1px solid #333';
            header.style.paddingBottom = '5px';

            const titleWrap = document.createElement('div');
            titleWrap.style.display = 'flex';
            titleWrap.style.alignItems = 'center';

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.className = 'sum-checkbox';
            checkbox.dataset.index = i;
            checkbox.style.marginRight = '10px';
            checkbox.style.width = '18px';
            checkbox.style.height = '18px';

            const title = document.createElement('strong');
            if (batch.type === "live") {
                title.textContent = `Live Turn (Msgs #${startNum} - #${endNum})`;
            } else {
                title.textContent = `Batch #${i + 1} (Msgs #${startNum} - #${endNum})`;
            }
            
            if (isConsolidated) {
                const badge = document.createElement('span');
                badge.className = 'sum-badge';
                badge.style.background = 'var(--accent)';
                badge.textContent = 'CONSOLIDATED';
                title.appendChild(badge);
            }
            if (batch.disabled) {
                const disBadge = document.createElement('span');
                disBadge.className = 'sum-badge';
                disBadge.style.background = '#0284c7';
                disBadge.textContent = 'SENDING AS RAW TEXT';
                title.appendChild(disBadge);
                card.style.opacity = '0.6';
            }

            titleWrap.appendChild(checkbox);
            titleWrap.appendChild(title);
            header.appendChild(titleWrap);

            const actionsWrap = document.createElement('div');
            if (isConsolidated) {
                const undoBtn = document.createElement('button');
                undoBtn.className = 'msg-btn';
                undoBtn.textContent = 'Undo Consolidation';
                undoBtn.onclick = () => undoConsolidation(i);
                actionsWrap.appendChild(undoBtn);
            } else {
                const usageText = document.createElement('span');
                usageText.style.fontSize = '0.8em';
                usageText.style.color = 'var(--text-dim)';
                usageText.textContent = batch.usage ? `In: ${batch.usage.prompt_tokens}, Out: ${batch.usage.completion_tokens}` : "";
                actionsWrap.appendChild(usageText);
            }
            header.appendChild(actionsWrap);

            const content = document.createElement('div');
            content.innerHTML = formatSummaryText(batch.content);
            const headerToggle = content.querySelector('.reasoning-header');
            if (headerToggle) {
                headerToggle.onclick = () => {
                    const c = content.querySelector('.reasoning-content');
                    const i = content.querySelector('.toggle-icon');
                    const isH = c.style.display === 'none';
                    c.style.display = isH ? 'block' : 'none';
                    i.textContent = isH ? '▲' : '▼';
                };
            }

            card.appendChild(header);
            card.appendChild(content);
            container.appendChild(card);
        });

        const sumModal = document.getElementById('summary-view-modal');
        if (sumModal) sumModal.style.display = 'block';
    }

    safeBind('view-all-summaries-btn', 'click', () => {
        renderMemoryManager();
    });

    safeBind('select-all-sums-btn', 'click', () => {
        const boxes = document.querySelectorAll('.sum-checkbox');
        const allChecked = Array.from(boxes).every(b => b.checked);
        boxes.forEach(b => b.checked = !allChecked);
    });

    safeBind('select-range-btn', 'click', () => {
        const startInput = document.getElementById('range-start');
        const endInput = document.getElementById('range-end');
        if (!startInput || !endInput) return;
        const start = parseInt(startInput.value);
        const end = parseInt(endInput.value);
        if (isNaN(start) || isNaN(end) || start > end || start < 1) {
            alert("Please enter a valid start and end batch number.");
            return;
        }
        const boxes = document.querySelectorAll('.sum-checkbox');
        boxes.forEach(b => {
            const idx = parseInt(b.dataset.index) + 1;
            b.checked = (idx >= start && idx <= end);
        });
    });

    safeBind('consolidate-selected-btn', 'click', async () => {
        const boxes = document.querySelectorAll('.sum-checkbox:checked');
        if (boxes.length < 2) {
            alert("Please select at least 2 summaries to consolidate.");
            return;
        }

        const indices = Array.from(boxes).map(b => parseInt(b.dataset.index)).sort((a,b) => a-b);
        for (let i = 1; i < indices.length; i++) {
            if (indices[i] !== indices[i-1] + 1) {
                alert("Please select a contiguous range of summaries.");
                return;
            }
        }

        const batchSizeEl = document.getElementById('cons-batch-size');
        const batchSize = batchSizeEl ? (parseInt(batchSizeEl.value) || 0) : 0;
        const btn = document.getElementById('consolidate-selected-btn');
        if (!btn) return;

        const ogText = btn.textContent;
        btn.textContent = "Consolidating...";
        btn.disabled = true;

        try {
            let chunks = [];
            if (batchSize > 0) {
                for (let i = 0; i < indices.length; i += batchSize) {
                    const chunk = indices.slice(i, i + batchSize);
                    if (chunk.length >= batchSize) {
                        chunks.push(chunk);
                    }
                }
            } else {
                chunks = [indices];
            }

            if (chunks.length === 0) {
                alert("Selection not large enough for the specified batch size.");
                btn.textContent = ogText; btn.disabled = false;
                return;
            }

            for (const chunk of chunks) {
                const res = await fetch('/consolidate_summaries', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({indices: chunk})
                });
                const d = await res.json();
                if (!d.success) {
                    alert("Error during consolidation: " + d.error);
                    break;
                }
            }

            await loadHistory();
            renderMemoryManager();
        } catch(e) {
            alert("Failed to consolidate.");
        }

        btn.textContent = ogText;
        btn.disabled = false;
    });

    safeBind('toggle-raw-btn', 'click', async () => {
        const boxes = document.querySelectorAll('.sum-checkbox:checked');
        if (boxes.length === 0) { alert("Select at least 1 summary."); return; }

        const indices = Array.from(boxes).map(b => parseInt(b.dataset.index));
        const firstIdx = indices[0];
        const isCurrentlyDisabled = summaries[firstIdx].disabled || false;
        const newState = !isCurrentlyDisabled;

        try {
            const res = await fetch('/toggle_summary_state', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({indices: indices, disabled: newState})
            });
            const d = await res.json();
            if (d.success) {
                await loadHistory();
                renderMemoryManager();
            }
        } catch(e) { alert("Failed to toggle state."); }
    });

    safeBind('regen-selected-sums-btn', 'click', async () => {
        const boxes = document.querySelectorAll('.sum-checkbox:checked');
        if (boxes.length === 0) { alert("Select at least 1 summary."); return; }

        const indices = Array.from(boxes).map(b => parseInt(b.dataset.index));
        if (!confirm(`Regenerate ${indices.length} summary batch(es)?`)) return;

        const btn = document.getElementById('regen-selected-sums-btn');
        const ogText = btn.textContent;
        btn.textContent = "Regenerating..."; btn.disabled = true;

        try {
            const res = await fetch('/force_summarize', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({mode: 'batches', batch_indices: indices})
            });
            const d = await res.json();
            if (d.success) {
                await loadHistory();
                renderMemoryManager();
            } else {
                alert("Error: " + d.error);
            }
        } catch(e) { alert("Failed to regenerate."); }
        btn.textContent = ogText; btn.disabled = false;
    });

    window.undoConsolidation = async (index) => {
        if (!confirm("Revert this consolidation back to its original summaries?")) return;
        try {
            const res = await fetch('/undo_consolidation', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({index: index})
            });
            const d = await res.json();
            if (d.success) {
                await loadHistory();
                renderMemoryManager();
            } else {
                alert("Error: " + d.error);
            }
        } catch(e) {
            alert("Failed to undo consolidation.");
        }
    };

    function openLargeEditor(index, type) {
        if (!editTextarea || !editOverlay) return;
        activeEditIndex = index;
        activeEditType = type;
        if (type === 'message') editTextarea.value = chatHistory[index].content.replace('__IMG_JSON__', '');
        else if (type === 'prompt') fetch('/get_settings').then(r=>r.json()).then(d => editTextarea.value = d.main_prompt);
        else if (type === 'visual') editTextarea.value = currentVisualMemory;
        else if (type === 'lorebook') fetch('/get_lorebook').then(r=>r.json()).then(d => editTextarea.value = d.text);
        editOverlay.style.display = 'block';
    }

    safeBind('save-large-edit', 'click', async () => {
        if (!editTextarea || !editOverlay) return;
        const val = editTextarea.value;
        if (activeEditType === 'message') {
            chatHistory[activeEditIndex].content = val;
            diffModeMsgs.delete(activeEditIndex);
            await saveHistory();
            renderChat();
        } else if (activeEditType === 'prompt') {
            await fetch('/save_settings', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({main_prompt:val}) });
        } else if (activeEditType === 'visual') {
            await fetch('/update_visual_memory', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({memory:val}) });
            currentVisualMemory = val;
            if(vmTextarea) vmTextarea.value = val;
            await loadHistory();
        } else if (activeEditType === 'lorebook') {
            await fetch('/save_lorebook', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text:val}) });
            if (confirm("Lorebook saved. Do you want to rebuild the vector index now to apply these changes? (This uses the Venice Embeddings API)")) {
                const btn = document.getElementById('rebuild-index-btn');
                if (btn) {
                    const ogText = btn.textContent;
                    btn.textContent = "Processing Embeddings..."; btn.disabled = true;
                    try {
                        const res = await fetch('/rebuild_index', { method: 'POST' });
                        const d = await res.json();
                        alert(d.message);
                    } catch(e) { alert("Rebuild failed."); }
                    btn.textContent = ogText; btn.disabled = false;
                }
            }
        }
        editOverlay.style.display = 'none';
    });

    safeBind('cancel-large-edit', 'click', () => { if(editOverlay) editOverlay.style.display = 'none'; });

    async function handleRegen(idx, role) {
        if (!confirm("Regenerate? Future messages will be deleted.")) return;
        diffModeMsgs.clear();
        if (role === 'user') {
            const txt = chatHistory[idx].content;
            chatHistory = chatHistory.slice(0, idx);
            if (userInput) { userInput.value = txt; autoExpandTextarea(userInput); }
            await saveHistory(); renderChat();
        } else {
            chatHistory = chatHistory.slice(0, idx);
            await saveHistory(); renderChat(); triggerGen("");
        }
    }

    async function handleBranch(idx) {
        if (!confirm("Branch conversation from this point into a new chat?")) return;
        try {
            const res = await fetch('/branch_chat', {
                method: 'POST', headers: {'Content-Type':'application/json'},
                body: JSON.stringify({index: idx})
            });
            const d = await res.json();
            if (d.success) {
                await loadSidebar();
                await loadChat(d.new_filename);
            } else {
                alert("Error: " + d.error);
            }
        } catch(e) {
            alert("Failed to branch chat.");
        }
    }

    async function saveHistory() {
        await fetch('/update_history', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({history:chatHistory}) });
    }

    function isCurrentModelVisionCapable() {
        if (!availableModels) return false;

        const modelSelect = document.getElementById('set-model-chat');
        let currentModelId = null;

        // 1. Try to get it from the settings modal dropdown (even if hidden)
        if (modelSelect && modelSelect.value) {
            currentModelId = modelSelect.value;
        } 
        // 2. Fallback: check chat history for the last model used
        else if (chatHistory.length > 0) {
            for (let i = chatHistory.length - 1; i >= 0; i--) {
                if (chatHistory[i].model) {
                    currentModelId = chatHistory[i].model;
                    break;
                }
            }
        }

        if (!currentModelId) return false;

        for (const group in availableModels) {
            const model = availableModels[group].find(m => m.id === currentModelId);
            if (model && model.vision) return true;
        }
        return false;
    }

    function updateAttachmentButtonVisibility() {
        const btn = document.getElementById('attachBtn');
        if (!btn) return;
        if (isCurrentModelVisionCapable()) {
            btn.style.display = 'block';
        } else {
            btn.style.display = 'none';
            clearAttachedImage();
        }
    }

    function clearAttachedImage() {
        attachedImageBase64 = null;
        const upload = document.getElementById('image-upload');
        if (upload) upload.value = '';
        const container = document.getElementById('image-preview-container');
        if (container) container.style.display = 'none';
    }

    safeBind('attachBtn', 'click', () => {
        document.getElementById('image-upload').click();
    });

    safeBind('image-upload', 'change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                let width = img.width;
                let height = img.height;
                const MAX_SIZE = 1024;

                if (width > height) {
                    if (width > MAX_SIZE) {
                        height *= MAX_SIZE / width;
                        width = MAX_SIZE;
                    }
                } else {
                    if (height > MAX_SIZE) {
                        width *= MAX_SIZE / height;
                        height = MAX_SIZE;
                    }
                }
                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, width, height);

                attachedImageBase64 = canvas.toDataURL('image/jpeg', 0.85);

                const preview = document.getElementById('image-preview');
                const container = document.getElementById('image-preview-container');
                if (preview && container) {
                    preview.src = attachedImageBase64;
                    container.style.display = 'flex';
                }
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    });

    safeBind('remove-image-btn', 'click', () => {
        clearAttachedImage();
    });

    async function sendMessage() {
        if (!userInput) return;
        const txt = userInput.value.trim();
        if(!txt && !attachedImageBase64) return;
        if(isGenerating) return;

        userInput.value = '';
        autoExpandTextarea(userInput);

        let messagePayload = txt;
        let customModelOverride = null;
        const pipeOverrideEl = document.getElementById('pipeline-model-override');
        if (pipeOverrideEl && pipeOverrideEl.style.display !== 'none' && pipeOverrideEl.value) {
            customModelOverride = pipeOverrideEl.value;
        }
        if (attachedImageBase64) {
            const highResToggle = document.getElementById('set-vision-high-res');
            const isHigh = highResToggle ? highResToggle.checked : true;
            messagePayload = [
                { "type": "text", "text": txt || "What is in this image?" },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": attachedImageBase64,
                        "detail": isHigh ? "high" : "low"
                    }
                }
            ];
        }

        clearAttachedImage();

        if (window.isArenaMode) {
            sendArenaMessage(messagePayload);
            return;
        }

        const cb = document.getElementById('compare-mode-cb');
        if (isComparing) {
            triggerCompareGen(messagePayload, true);
            return;
        } else if (cb && cb.checked) {
            triggerCompareGen(messagePayload, false);
            return;
        }

        const sendTime = new Date().toISOString();
        const msgObj = {role:'user', content:messagePayload, timestamp: sendTime};
        if (txt === lastWfmText && lastWfmUsage) {
            msgObj.usage = lastWfmUsage;
        }
        chatHistory.push(msgObj);
        renderChat();
        triggerGen(messagePayload, customModelOverride, sendTime);
    }

    let isComparing = false;
    let compareHistoryA = [];
    let compareHistoryB = [];
    let compareContainerEl = null;
    let colA_chat = null;
    let colB_chat = null;
    let colA_stats = null;
    let colB_stats = null;
    let compUsageA = { prompt: 0, comp: 0 };
    let compUsageB = { prompt: 0, comp: 0 };

    async function triggerCompareGen(msg, isContinuation) {
        lastWfmText = ""; lastWfmUsage = null;
        isGenerating = true;

        if (!isContinuation) {
            compareHistoryA = []; compareHistoryB = [];
            compUsageA = {prompt:0, comp:0}; compUsageB = {prompt:0, comp:0};

            compareContainerEl = document.createElement('div');
            compareContainerEl.className = 'compare-container';

            compareContainerEl.innerHTML = `
                <div class="compare-col">
                    <h4 style="color:var(--accent);">Version A: Current Summaries</h4>
                    <div class="compare-chatbox" id="comp-chat-a"></div>
                    <div class="compare-stats" id="comp-stats-a">Usage: Pending...</div>
                    <button class="keep-btn" id="keep-btn-a" style="background:var(--accent); display:none;">Choose Version A</button>
                </div>
                <div class="compare-col">
                    <h4 style="color:var(--purple);">Version B: Original Summaries</h4>
                    <div class="compare-chatbox" id="comp-chat-b"></div>
                    <div class="compare-stats" id="comp-stats-b">Usage: Pending...</div>
                    <button class="keep-btn" id="keep-btn-b" style="background:var(--purple); display:none;">Choose Version B</button>
                </div>
            `;

            if (chatbox) chatbox.appendChild(compareContainerEl);
            colA_chat = compareContainerEl.querySelector('#comp-chat-a');
            colB_chat = compareContainerEl.querySelector('#comp-chat-b');
            colA_stats = compareContainerEl.querySelector('#comp-stats-a');
            colB_stats = compareContainerEl.querySelector('#comp-stats-b');

            compareContainerEl.querySelector('#keep-btn-a').onclick = () => resolveCompare('A');
            compareContainerEl.querySelector('#keep-btn-b').onclick = () => resolveCompare('B');
        }

        compareContainerEl.querySelector('#keep-btn-a').style.display = 'none';
        compareContainerEl.querySelector('#keep-btn-b').style.display = 'none';

        const sendTime = new Date().toISOString();
        compareHistoryA.push({role: 'user', content: msg, timestamp: sendTime});
        compareHistoryB.push({role: 'user', content: msg, timestamp: sendTime});

        const userDivA = document.createElement('div'); userDivA.className = 'message user'; 
        const userDivB = document.createElement('div'); userDivB.className = 'message user'; 
        let displayMsg = msg;
        if (Array.isArray(msg)) {
            displayMsg = msg.find(p=>p.type==='text')?.text || "[Image Attached]";
        }
        userDivA.textContent = displayMsg;
        userDivB.textContent = displayMsg;
        if (colA_chat) colA_chat.appendChild(userDivA); 
        if (colB_chat) colB_chat.appendChild(userDivB);

        const astDivA = document.createElement('div'); astDivA.className = 'message assistant'; astDivA.innerHTML = '<i>Thinking...</i>';
        const astDivB = document.createElement('div'); astDivB.className = 'message assistant'; astDivB.innerHTML = '<i>Thinking...</i>';
        if (colA_chat) colA_chat.appendChild(astDivA); 
        if (colB_chat) colB_chat.appendChild(astDivB);
        if (chatbox) chatbox.scrollTop = chatbox.scrollHeight;

        let fullA = ""; let fullB = "";
        let finalUsageA = null; let finalUsageB = null;

        try {
            const res = await fetch('/compare_chat', { 
                method:'POST', headers:{'Content-Type':'application/json'}, 
                body:JSON.stringify({history_A: compareHistoryA, history_B: compareHistoryB}) 
            });
            const reader = res.body.getReader();
            const dec = new TextDecoder();
            while(true){
                const {done, value} = await reader.read();
                if(done) break;
                const chunk = dec.decode(value);

                const isAtBottom = chatbox.scrollHeight - chatbox.scrollTop <= chatbox.clientHeight + 100;

                chunk.split('\n\n').forEach(l=>{
                   if(l.startsWith('data: ')){
                       try { 
                           const d = JSON.parse(l.substring(6)); 
                           if(d.A) {
                               if (d.A.content) fullA += d.A.content;
                               if (d.A.usage) finalUsageA = d.A.usage;
                               if (d.A.balance) updateBalanceDisplay(d.A.balance);
                               astDivA.innerHTML = typeof marked !== 'undefined' ? marked.parse(fullA) : fullA;
                           }
                           if(d.B) {
                               if (d.B.content) fullB += d.B.content;
                               if (d.B.usage) finalUsageB = d.B.usage;
                               if (d.B.balance) updateBalanceDisplay(d.B.balance);
                               astDivB.innerHTML = typeof marked !== 'undefined' ? marked.parse(fullB) : fullB;
                           }
                           if (colA_chat) colA_chat.scrollTop = colA_chat.scrollHeight;
                           if (colB_chat) colB_chat.scrollTop = colB_chat.scrollHeight;
                       } catch(e){}
                   } 
                });

                if (isAtBottom) {
                    chatbox.scrollTop = chatbox.scrollHeight;
                }
            }

            if (finalUsageA) { compUsageA.prompt += finalUsageA.prompt_tokens; compUsageA.comp += finalUsageA.completion_tokens; }
            if (finalUsageB) { compUsageB.prompt += finalUsageB.prompt_tokens; compUsageB.comp += finalUsageB.completion_tokens; }

            const aEntry = {role: 'assistant', content: fullA, timestamp: new Date().toISOString()};
            if (finalUsageA) aEntry.usage = finalUsageA;
            compareHistoryA.push(aEntry);

            const bEntry = {role: 'assistant', content: fullB, timestamp: new Date().toISOString()};
            if (finalUsageB) bEntry.usage = finalUsageB;
            compareHistoryB.push(bEntry);
            
            // Note: Removed auto-scroll here to respect user position
        } catch(e) { 
            astDivA.textContent = "Error"; astDivB.textContent = "Error"; 
        } finally {
            isGenerating = false;
            isComparing = true;
        }
    }

    async function resolveCompare(choice) {
        const chosen_history = choice === 'A' ? compareHistoryA : compareHistoryB;

        if (compareContainerEl) compareContainerEl.remove();
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message assistant system-msg';
        loadingDiv.innerHTML = `<i>Saving choice ${choice} and consolidating branch...</i>`;
        if (chatbox) chatbox.appendChild(loadingDiv);

        try {
            await fetch('/resolve_comparison', {
                method: 'POST', headers:{'Content-Type':'application/json'},
                body: JSON.stringify({
                    choice: choice,
                    chosen_history: chosen_history
                })
            });
            isComparing = false;
            compareHistoryA = []; compareHistoryB = [];
            loadingDiv.remove();
            await loadHistory();
        } catch(e) {
            alert("Failed to save choice.");
            loadingDiv.remove();
        }
    }

    async function handleRegenWithModel(idx) {
        try {
            const mData = availableModels || await fetch('/venice_models').then(r => r.json());
            const select = document.getElementById('regen-model-select');
            if(!select) return;
            select.innerHTML = '';
            for (const group in mData) {
                const optgroup = document.createElement('optgroup');
                optgroup.label = group;
                mData[group].forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m.id;
                    opt.textContent = m.name;
                    optgroup.appendChild(opt);
                });
                select.appendChild(optgroup);
            }
            const modal = document.getElementById('regen-model-modal');
            modal.style.display = 'block';

            document.getElementById('regen-model-cancel').onclick = () => modal.style.display = 'none';
            document.getElementById('regen-model-confirm').onclick = async () => {
                if (!confirm("Regenerate? Future messages will be deleted.")) return;
                const selectedModel = select.value;
                modal.style.display = 'none';
                diffModeMsgs.clear();
                chatHistory = chatHistory.slice(0, idx);
                await saveHistory(); 
                renderChat(); 
                triggerGen("", selectedModel);
            };
        } catch(e) { alert("Failed to load models."); }
    }

    // --- NAVIGATION WIDGET LOGIC ---
    safeBind('nav-top', 'click', () => {
        chatbox.scrollTo({ top: 0, behavior: 'smooth' });
    });

    safeBind('nav-bottom', 'click', () => {
        chatbox.scrollTo({ top: chatbox.scrollHeight, behavior: 'smooth' });
    });

    safeBind('nav-up', 'click', () => {
        // Find current message in view and jump to previous
        const msgs = document.querySelectorAll('.message');
        for (let i = msgs.length - 1; i >= 0; i--) {
            const rect = msgs[i].getBoundingClientRect();
            if (rect.top < 0) {
                msgs[i].scrollIntoView({ behavior: 'smooth' });
                break;
            }
        }
    });

    safeBind('nav-down', 'click', () => {
        const msgs = document.querySelectorAll('.message');
        for (let i = 0; i < msgs.length; i++) {
            const rect = msgs[i].getBoundingClientRect();
            if (rect.top > chatbox.clientHeight) {
                msgs[i].scrollIntoView({ behavior: 'smooth' });
                break;
            }
        }
    });

    // --- AI MEMORY LOGIC ---
    safeBind('open-memory-btn', 'click', () => {
        const modal = document.getElementById('memory-modal');
        if (modal) {
            modal.style.display = 'block';
            renderMemoryManagerUI();
        }
    });

    safeBind('close-memory-x', 'click', () => {
        const modal = document.getElementById('memory-modal');
        if (modal) modal.style.display = 'none';
    });

    function renderMemoryManagerUI() {
        const contentTa = document.getElementById('memory-content-text');
        const logsDiv = document.getElementById('memory-logs-list');
        if (contentTa) contentTa.value = currentSelfMemory;
        if (logsDiv) {
            logsDiv.innerHTML = '';
            if (memoryLogs.length === 0) {
                logsDiv.innerHTML = '<div style="color:#666; font-style:italic; text-align:center; padding:20px;">No change history found.</div>';
                return;
            }
            memoryLogs.forEach(log => {
                const item = document.createElement('div');
                item.style.cssText = "padding:8px; border-bottom:1px solid #222; display:flex; justify-content:space-between; align-items:flex-start; gap:10px;";
                const date = new Date(log.timestamp).toLocaleString();
                item.innerHTML = `
                    <div style="flex:1;">
                        <div style="color:var(--text-dim); font-size:0.8em;">${date}</div>
                        <div style="color:#ccc;">${log.change}</div>
                    </div>
                    <div style="font-size:0.75em; color:var(--primary); background:rgba(37,99,235,0.1); padding:2px 6px; border-radius:4px;">${log.model}</div>
                `;
                logsDiv.appendChild(item);
            });
        }
    }

    safeBind('save-memory-btn', 'click', async () => {
        const val = document.getElementById('memory-content-text').value;
        const btn = document.getElementById('save-memory-btn');
        const ogText = btn.textContent;
        btn.textContent = "Saving..."; btn.disabled = true;

        try {
            await fetch('/update_visual_memory', { // We reuse the update route but specify self_memory
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ memory: val, type: 'self' }) // Add type to distinguish
            });
            currentSelfMemory = val;
            alert("Internal Memory updated manually.");
            await loadHistory();
        } catch(e) { alert("Failed to save memory."); }
        
        btn.textContent = ogText; btn.disabled = false;
    });

    async function triggerGen(msg, customModel = null, timestamp = null) {
        lastWfmText = "";
        lastWfmUsage = null;
        isGenerating = true;

        const statusMsg = document.getElementById('status-msg');
        if (statusMsg) statusMsg.innerHTML = '';

        const div = document.createElement('div'); div.className = 'message assistant'; div.textContent = '...';
        if (chatbox) {
            chatbox.appendChild(div); 
            chatbox.scrollTop = chatbox.scrollHeight;
        }
        let full = "";
        let reasoning = "";
        try {
            const payload = {message: msg};
            if (customModel) payload.custom_model = customModel;
            if (timestamp) payload.timestamp = timestamp;
            if (isAuditMode && auditContextSelection.batches.length > 0) {
                payload.audit_context = auditContextSelection;
            }
            
            const res = await fetch('/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
            const reader = res.body.getReader();
            const dec = new TextDecoder();

            let rBlock = null;
            let sBlock = null;
            let contentDiv = null;
            let liveSummary = "";
            let hasClearedPlaceholder = false;
            let inThinkTag = false;
            let streamBuffer = "";

            while(true){
                const {done, value} = await reader.read();
                if(done) break;
                streamBuffer += dec.decode(value, { stream: true });
                const events = streamBuffer.split('\n\n');
                streamBuffer = events.pop();

                // Check if user is at bottom before updating content
                const isAtBottom = chatbox.scrollHeight - chatbox.scrollTop <= chatbox.clientHeight + 100;

                events.forEach(l=>{
                   if(l.startsWith('data: ')){
                       try{ 
                           const d = JSON.parse(l.substring(6)); 

                           // Check if user is at bottom before updating content to handle sticky scroll
                           const isAtBottom = chatbox.scrollHeight - chatbox.scrollTop <= chatbox.clientHeight + 100;

                           if(d.status) {
                               if (statusMsg) {
                                   statusMsg.innerHTML = `<div class="spinner"></div> <span style="color:var(--purple)">${d.status}</span>`;
                               }
                           }

                           if(d.reasoning) {
                               if (statusMsg) statusMsg.innerHTML = '';
                               if (!hasClearedPlaceholder) { div.innerHTML = ''; hasClearedPlaceholder = true; }
                               if(!rBlock) {
                                   rBlock = document.createElement('div');
                                   rBlock.className = 'reasoning-block';
                                   rBlock.innerHTML = `
                                       <div class="reasoning-header">
                                           <span><i style="margin-right:5px;">💭</i> Thought Process</span>
                                           <span class="toggle-icon">▲</span>
                                       </div>
                                       <div class="reasoning-content" style="display:block;"></div>
                                   `;
                                   div.appendChild(rBlock);
                               }
                               reasoning += d.reasoning;
                               rBlock.querySelector('.reasoning-content').textContent = reasoning;
                               
                               if (isAtBottom) chatbox.scrollTop = chatbox.scrollHeight;
                           }

                           if(d.live_summary) {
                               if (statusMsg) statusMsg.innerHTML = '';
                               if (!hasClearedPlaceholder) { div.innerHTML = ''; hasClearedPlaceholder = true; }
                               if(!sBlock) {
                                   sBlock = document.createElement('div');
                                   sBlock.className = 'summary-block';
                                   sBlock.innerHTML = `
                                       <div class="summary-header">
                                           <span><i style="margin-right:5px;">📝</i> Live Inline Summary</span>
                                           <span class="toggle-icon-sum">▲</span>
                                       </div>
                                       <div class="summary-content" style="display:block;"></div>
                                   `;
                                   sBlock.querySelector('.summary-header').onclick = () => {
                                       const c = sBlock.querySelector('.summary-content');
                                       const i = sBlock.querySelector('.toggle-icon-sum');
                                       const isH = c.style.display === 'none';
                                       c.style.display = isH ? 'block' : 'none';
                                       i.textContent = isH ? '▲' : '▼';
                                   };
                                   if (rBlock && rBlock.nextSibling) {
                                       div.insertBefore(sBlock, rBlock.nextSibling);
                                   } else {
                                       div.appendChild(sBlock);
                                   }
                               }
                               liveSummary += d.live_summary;
                               sBlock.querySelector('.summary-content').textContent = liveSummary;
                               
                               if (isAtBottom) chatbox.scrollTop = chatbox.scrollHeight;
                           }

                           if(d.content) {
                               if (statusMsg) statusMsg.innerHTML = '';
                               if (!hasClearedPlaceholder) { div.innerHTML = ''; hasClearedPlaceholder = true; }
                               
                               // Handle cases where reasoning is delivered inside <think> tags in the main content
                               if (d.content.includes('<think>')) {
                                   inThinkTag = true;
                                   const parts = d.content.split('<think>');
                                   
                                   if (parts[0]) {
                                       full += parts[0];
                                       if(!contentDiv) {
                                           contentDiv = document.createElement('div');
                                           contentDiv.style.cssText = "white-space:pre-wrap; font-family:inherit;";
                                           div.appendChild(contentDiv);
                                       }
                                       contentDiv.innerHTML = typeof marked !== 'undefined' ? marked.parse(full) : full;
                                   }
                                   
                                   if (!rBlock) {
                                       rBlock = document.createElement('div');
                                       rBlock.className = 'reasoning-block';
                                       rBlock.innerHTML = `
                                           <div class="reasoning-header">
                                               <span><i style="margin-right:5px;">💭</i> Thought Process</span>
                                               <span class="toggle-icon">▲</span>
                                           </div>
                                           <div class="reasoning-content" style="display:block;"></div>
                                       `;
                                       rBlock.querySelector('.reasoning-header').onclick = () => {
                                           const c = rBlock.querySelector('.reasoning-content');
                                           const i = rBlock.querySelector('.toggle-icon');
                                           const isH = c.style.display === 'none';
                                           c.style.display = isH ? 'block' : 'none';
                                           i.textContent = isH ? '▲' : '▼';
                                       };
                                       div.appendChild(rBlock); // Append instead of insertBefore so it flows naturally
                                   }
                                   
                                   if (parts[1]) {
                                       if (parts[1].includes('</think>')) {
                                           inThinkTag = false;
                                           const subParts = parts[1].split('</think>');
                                           reasoning += subParts[0];
                                           rBlock.querySelector('.reasoning-content').textContent = reasoning;
                                           if (subParts[1]) {
                                               full += subParts[1];
                                               if(!contentDiv) {
                                                   contentDiv = document.createElement('div');
                                                   contentDiv.style.cssText = "white-space:pre-wrap; font-family:inherit;";
                                                   div.appendChild(contentDiv);
                                               }
                                               contentDiv.innerHTML = typeof marked !== 'undefined' ? marked.parse(full) : full;
                                           }
                                       } else {
                                           reasoning += parts[1];
                                           rBlock.querySelector('.reasoning-content').textContent = reasoning;
                                       }
                                   }
                               } else if (d.content.includes('</think>')) {
                                   inThinkTag = false;
                                   const parts = d.content.split('</think>');
                                   reasoning += parts[0];
                                   if (rBlock) rBlock.querySelector('.reasoning-content').textContent = reasoning;
                                   if (parts[1]) {
                                       full += parts[1];
                                       if(!contentDiv) {
                                           contentDiv = document.createElement('div');
                                           contentDiv.style.cssText = "white-space:pre-wrap; font-family:inherit;";
                                           div.appendChild(contentDiv);
                                       }
                                       contentDiv.innerHTML = typeof marked !== 'undefined' ? marked.parse(full) : full;
                                   }
                               } else if (inThinkTag) {
                                    // We are inside a <think> block
                                    reasoning += d.content;
                                    if (rBlock) rBlock.querySelector('.reasoning-content').textContent = reasoning;
                               } else {
                                   // Normal content delivery
                                   if(!contentDiv) {
                                       contentDiv = document.createElement('div');
                                       contentDiv.style.cssText = "white-space:pre-wrap; font-family:inherit;";
                                       div.appendChild(contentDiv);
                                       
                                       // Collapse reasoning/summary if we just created the content block
                                       if(rBlock && rBlock.querySelector('.reasoning-content').style.display !== 'none') {
                                           rBlock.querySelector('.reasoning-content').style.display = 'none';
                                           rBlock.querySelector('.toggle-icon').textContent = '▼';
                                       }
                                       if(sBlock && sBlock.querySelector('.summary-content').style.display !== 'none') {
                                           sBlock.querySelector('.summary-content').style.display = 'none';
                                           sBlock.querySelector('.toggle-icon-sum').textContent = '▼';
                                       }
                                   }
                                   full += d.content;
                                   
                                   // [INJECTION] Hide blueprint tags during streaming
                                   let displayFull = full;
                                   displayFull = displayFull.replace(/\[(?:ADD_TO|REWRITE|EDIT)_BLUEPRINT\].*?(\[\/(?:ADD_TO|REWRITE|EDIT)_BLUEPRINT\]|$)/gs, '');
                                   
                                   contentDiv.innerHTML = typeof marked !== 'undefined' ? marked.parse(displayFull.trim()) : displayFull.trim();
                               }
                               if (d.blueprint_update) {
                                   const docEditor = document.getElementById('blueprint-editor');
                                   if (docEditor) {
                                       const oldBp = previousBlueprint;
                                       const newBp = d.blueprint_update;
                                       docEditor.value = newBp;
                                       updateBlueprintDiff(oldBp, newBp);
                                       previousBlueprint = newBp;
                                   }
                               }
                               if (d.tool_calls) {
                                   // Render a collapsible "Blueprint Tool Calls" block
                                   let tcBlock = div.querySelector('.tool-calls-block');
                                   if (!tcBlock) {
                                       tcBlock = document.createElement('div');
                                       tcBlock.className = 'tool-calls-block';
                                       tcBlock.style.cssText = 'margin-top:8px; border:1px solid #2d4a2d; border-radius:6px; overflow:hidden; font-size:0.85em;';
                                       tcBlock.innerHTML = `
                                           <div class="tool-calls-header" style="display:flex; justify-content:space-between; align-items:center; padding:6px 12px; background:#1a2e1a; cursor:pointer; user-select:none;">
                                               <span><i style="margin-right:5px;">🔧</i>Blueprint Tool Calls</span>
                                               <span class="toggle-icon-tc">▼</span>
                                           </div>
                                           <div class="tool-calls-content" style="display:none; padding:12px; background:#111; white-space:pre-wrap; font-family:monospace; color:#a3e635; overflow-x:auto;"></div>
                                       `;
                                       tcBlock.querySelector('.tool-calls-header').onclick = () => {
                                           const c = tcBlock.querySelector('.tool-calls-content');
                                           const ic = tcBlock.querySelector('.toggle-icon-tc');
                                           const isH = c.style.display === 'none';
                                           c.style.display = isH ? 'block' : 'none';
                                           ic.textContent = isH ? '▲' : '▼';
                                       };
                                       div.appendChild(tcBlock);
                                   }
                                   tcBlock.querySelector('.tool-calls-content').textContent = d.tool_calls;
                               }
                               if (d.content_overwrite) {
                                   full = d.content_overwrite;
                                   if (contentDiv) contentDiv.innerHTML = typeof marked !== 'undefined' ? marked.parse(full) : full;
                               }
                               
                               if (isAtBottom) chatbox.scrollTop = chatbox.scrollHeight;
                           }

                           // These pipeline/finalization events may arrive without d.content.
                           // They must be handled outside the if(d.content) block.
                           if (!d.content) {
                               if (d.blueprint_update) {
                                   const docEditor = document.getElementById('blueprint-editor');
                                   if (docEditor) {
                                       const oldBp = previousBlueprint;
                                       const newBp = d.blueprint_update;
                                       docEditor.value = newBp;
                                       updateBlueprintDiff(oldBp, newBp);
                                       previousBlueprint = newBp;
                                   }
                               }
                               if (d.tool_calls) {
                                   if (!hasClearedPlaceholder) {
                                       div.innerHTML = '';
                                       hasClearedPlaceholder = true;
                                   }
                                   let tcBlock = div.querySelector('.tool-calls-block');
                                   if (!tcBlock) {
                                       tcBlock = document.createElement('div');
                                       tcBlock.className = 'tool-calls-block';
                                       tcBlock.style.cssText = 'margin-top:8px; border:1px solid #2d4a2d; border-radius:6px; overflow:hidden; font-size:0.85em;';
                                       tcBlock.innerHTML = `
                                           <div class="tool-calls-header" style="display:flex; justify-content:space-between; align-items:center; padding:6px 12px; background:#1a2e1a; cursor:pointer; user-select:none;">
                                               <span><i style="margin-right:5px;">🔧</i>Blueprint Tool Calls</span>
                                               <span class="toggle-icon-tc">▼</span>
                                           </div>
                                           <div class="tool-calls-content" style="display:none; padding:12px; background:#111; white-space:pre-wrap; font-family:monospace; color:#a3e635; overflow-x:auto;"></div>
                                       `;
                                       tcBlock.querySelector('.tool-calls-header').onclick = () => {
                                           const c = tcBlock.querySelector('.tool-calls-content');
                                           const ic = tcBlock.querySelector('.toggle-icon-tc');
                                           const isH = c.style.display === 'none';
                                           c.style.display = isH ? 'block' : 'none';
                                           ic.textContent = isH ? '▲' : '▼';
                                       };
                                       div.appendChild(tcBlock);
                                   }
                                   tcBlock.querySelector('.tool-calls-content').textContent = d.tool_calls;
                               }
                               if (d.content_overwrite) {
                                   if (statusMsg) statusMsg.innerHTML = '';
                                   if (!hasClearedPlaceholder) {
                                       div.innerHTML = '';
                                       hasClearedPlaceholder = true;
                                   }
                                   full = d.content_overwrite;
                                   if (!contentDiv) {
                                       contentDiv = document.createElement('div');
                                       contentDiv.style.cssText = "white-space:pre-wrap; font-family:inherit;";
                                       div.insertBefore(contentDiv, div.firstChild);
                                   }
                                   contentDiv.innerHTML = typeof marked !== 'undefined' ? marked.parse(full) : full;
                               }
                               if (isAtBottom) chatbox.scrollTop = chatbox.scrollHeight;
                           }
                           if(d.error) {

                               if (statusMsg) statusMsg.innerHTML = '';
                               console.error("Server Error:", d.error);
                               full += "\n\n**System Error:** " + d.error;
                               if(contentDiv) contentDiv.innerHTML = typeof marked !== 'undefined' ? marked.parse(full) : full;
                               else div.textContent = full;
                           }

                           if(d.balance) updateBalanceDisplay(d.balance);
                           if(d.balance_refresh) updateBalanceDisplay(d.balance_refresh);

                           if(d.memory_update) {
                               currentSelfMemory = d.memory_update;
                               if (d.memory_logs) {
                                   memoryLogs = d.memory_logs;
                               }
                               
                               // If the modal is open, re-render it immediately
                               if (document.getElementById('memory-modal')?.style.display === 'block') {
                                   renderMemoryManagerUI();
                               }

                               if (statusMsg) {
                                   const toast = document.createElement('div');
                                   toast.style.cssText = "color:var(--accent); font-weight:bold; animation: fadeOut 3s forwards;";
                                   toast.innerHTML = "ðŸ§  AI updated its Internal Notepad";
                                   statusMsg.appendChild(toast);
                                   setTimeout(() => toast.remove(), 4000);
                               }
                           }
                       }catch(e){}
                   } 
                });

                if (isAtBottom) {
                    // We remove this to give user total control during stream
                    // chatbox.scrollTop = chatbox.scrollHeight;
                }
            }
            await loadHistory();
        } catch(e) { 
            if (statusMsg) statusMsg.innerHTML = '';
            div.textContent = "Error"; 
        }
        finally { isGenerating = false; }
    }

    async function handleImageBtn() {
        if(guidedModeEnabled) { 
            if (guidedModal) guidedModal.style.display='block'; 
            if (guidedInput) guidedInput.focus(); 
        }
        else triggerImageGen("");
    }

    async function triggerImageGen(g, debugMode = false) {
        if(isGenerating) return; 
        isGenerating = true; 
        if (guidedModal) guidedModal.style.display='none';
        const genBtn = document.getElementById('genImgBtn');
        let originalBtnText = "Generate Image";
        if (genBtn) {
            originalBtnText = genBtn.textContent;
            genBtn.textContent = "Generating Image..."; genBtn.disabled = true;
        }
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message assistant system-msg'; loadingDiv.id = 'temp-image-loading';
        loadingDiv.innerHTML = `<i>Painting image... please wait.</i>`;
        if (chatbox) {
            chatbox.appendChild(loadingDiv); chatbox.scrollTop = chatbox.scrollHeight;
        }
        try {
            const res = await fetch('/generate_image', {
                method:'POST', 
                headers:{'Content-Type':'application/json'}, 
                body:JSON.stringify({guidance:g, debug_mode: debugMode})
            });
            const d = await res.json();

            if(res.ok) { 
                await loadHistory(); 
                if (debugMode && d.debug) showDebugModal(d.debug);
            } 
            else { 
                if (debugMode && d.debug) showDebugModal(d.debug);
                alert("Gen Failed: " + (d.error || "Unknown Error")); 
            }
        } catch(e) { alert("API Error: " + e.message); }
        finally { 
            isGenerating = false; 
            if (genBtn) { genBtn.textContent = originalBtnText; genBtn.disabled = false; }
            const tempMsg = document.getElementById('temp-image-loading'); if(tempMsg) tempMsg.remove();
        }
    }

    async function loadSidebar() {
        try {
            const res = await fetch('/sidebar_data');
            const data = await res.json();
            const list = document.getElementById('chat-list');
            if (!list) return;
            list.innerHTML = '';
            
            const normalChats = data.chats.filter(f => !f.endsWith('.audit.json'));
            const auditChats = data.chats.filter(f => f.endsWith('.audit.json'));
            
            normalChats.forEach(file => {
                const li = document.createElement('li'); li.className = 'chat-item';
                if (file === data.active_chat) li.classList.add('active-chat');
                li.innerHTML = `<span class="chat-name">${file.replace('.json','').replace(/_/g,' ')}</span>
                                <div><button class="icon-btn r-btn">&#9998;</button><button class="icon-btn d-btn">&#128465;</button></div>`;
                li.onclick = (e) => { if(!e.target.closest('.icon-btn')) loadChat(file); };
                li.querySelector('.r-btn').onclick = () => renameChat(file);
                li.querySelector('.d-btn').onclick = () => deleteChat(file);
                list.appendChild(li);
                
                // Check if this chat has an audit child
                const auditFile = file.replace('.json', '.audit.json');
                if (auditChats.includes(auditFile)) {
                    const ali = document.createElement('li');
                    ali.className = 'chat-item audit-chat-item';
                    if (auditFile === data.active_chat) ali.classList.add('active-chat');
                    ali.innerHTML = `<span class="chat-name">â†³ ðŸ•µï¸ Audit: ${file.replace('.json','').replace(/_/g,' ')}</span>
                                    <div><button class="icon-btn d-btn">ðŸ—‘</button></div>`;
                    ali.onclick = (e) => { if(!e.target.closest('.icon-btn')) loadChat(auditFile); };
                    ali.querySelector('.d-btn').onclick = () => deleteChat(auditFile);
                    list.appendChild(ali);
                }
            });
            
            // Render orphaned audit chats just in case parent was deleted
            auditChats.forEach(aFile => {
                const parentExpected = aFile.replace('.audit.json', '.json');
                if (!normalChats.includes(parentExpected)) {
                    const ali = document.createElement('li');
                    ali.className = 'chat-item audit-chat-item';
                    if (aFile === data.active_chat) ali.classList.add('active-chat');
                    ali.innerHTML = `<span class="chat-name">â†³ ðŸ•µï¸ Orphaned Audit: ${aFile.replace('.audit.json','').replace(/_/g,' ')}</span>
                                    <div><button class="icon-btn d-btn">ðŸ—‘</button></div>`;
                    ali.onclick = (e) => { if(!e.target.closest('.icon-btn')) loadChat(aFile); };
                    ali.querySelector('.d-btn').onclick = () => deleteChat(aFile);
                    list.appendChild(ali);
                }
            });
            
        } catch(e) {
            console.error("Sidebar load failed:", e);
        }
    }

    let activeRenameFile = null;

    window.renameChat = async (file) => {
        activeRenameFile = file;
        const modal = document.getElementById('rename-modal');
        const input = document.getElementById('manual-rename-input');
        const title = document.getElementById('rename-modal-title');
        const stats = document.getElementById('rename-stats');
        const startIn = document.getElementById('rename-start');
        const endIn = document.getElementById('rename-end');

        if (!modal || !input) return;

        title.textContent = `Rename: ${file.replace('.json', '').replace(/_/g, ' ')}`;
        input.value = file.replace('.json', '').replace(/_/g, ' ');

        // If this is the active chat, we can show stats
        const activeMeta = await fetch('/sidebar_data').then(r => r.json());
        if (activeMeta.active_chat === file) {
            const textMsgCount = chatHistory.filter(m => m.role !== 'system' && !(typeof m.content === 'string' && m.content.startsWith('__IMG_JSON__'))).length;
            if (stats) stats.textContent = `${textMsgCount} messages total`;
            if (startIn) { 
                startIn.value = 1; 
                startIn.max = textMsgCount; 
            }
            if (endIn) { 
                endIn.value = Math.min(15, textMsgCount); 
                endIn.max = textMsgCount; 
            }
        } else {
            if (stats) stats.textContent = "";
            if (startIn) startIn.value = 1;
            if (endIn) endIn.value = 15;
        }

        modal.style.display = 'block';
        input.focus();
    };

    safeBind('close-rename-x', 'click', () => {
        document.getElementById('rename-modal').style.display = 'none';
        activeRenameFile = null;
    });

    safeBind('close-logic-view-x', 'click', () => {
        const modal = document.getElementById('logic-view-modal');
        if (modal) modal.style.display = 'none';
    });

    safeBind('manual-rename-submit', 'click', async () => {
        const newName = document.getElementById('manual-rename-input').value.trim();
        if (!newName || !activeRenameFile) return;

        try {
            const res = await fetch('/rename_chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ old: activeRenameFile, new: newName })
            });
            const d = await res.json();
            if (d.success) {
                document.getElementById('rename-modal').style.display = 'none';
                activeRenameFile = null;
                await loadSidebar();
            }
        } catch (e) {
            alert("Failed to rename chat.");
        }
    });

    safeBind('ai-rename-submit', 'click', async () => {
        const btn = document.getElementById('ai-rename-submit');
        const modelSelect = document.getElementById('rename-model-select');
        const startIn = document.getElementById('rename-start');
        const endIn = document.getElementById('rename-end');

        if (!btn || !modelSelect || !activeRenameFile) return;

        // Ensure we are operating on the chat being renamed
        const activeMeta = await fetch('/sidebar_data').then(r => r.json());
        if (activeMeta.active_chat !== activeRenameFile) {
            if (confirm("AI Renaming requires loading the chat first. Load this chat now?")) {
                await loadChat(activeRenameFile);
                // After loading, the range might need adjustment
                const textMsgCount = chatHistory.filter(m => m.role !== 'system' && !(typeof m.content === 'string' && m.content.startsWith('__IMG_JSON__'))).length;
                if (startIn) startIn.value = 1;
                if (endIn) endIn.value = Math.min(15, textMsgCount);
            } else {
                return;
            }
        }

        const ogText = btn.textContent;
        btn.textContent = "AI is thinking..."; btn.disabled = true;

        try {
            // Calculate range indices based on msg count
            const textMsgs = chatHistory.filter(m => m.role !== 'system' && !(typeof m.content === 'string' && m.content.startsWith('__IMG_JSON__')));
            const startNum = parseInt(startIn.value) || 1;
            const endNum = parseInt(endIn.value) || 15;
            
            // We'll pass the range to generate_chat_title (we need to update main.py for this)
            const res = await fetch('/generate_chat_title', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    model: modelSelect.value,
                    range: { start: startNum, end: endNum }
                })
            });
            const d = await res.json();

            if (d.success) {
                const finalName = prompt("AI suggested title. Click OK to apply or edit it below:", d.title);
                if (finalName) {
                    const renRes = await fetch('/rename_chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ old: activeRenameFile, new: finalName })
                    });
                    if (renRes.ok) {
                        document.getElementById('rename-modal').style.display = 'none';
                        activeRenameFile = null;
                        await loadSidebar();
                    }
                }
            } else {
                alert("Rename Error: " + d.error);
            }
        } catch (e) {
            console.error(e);
            alert("Failed to auto-rename.");
        }
        btn.textContent = ogText; btn.disabled = false;
    });
    window.deleteChat = async (file) => {
        if(confirm("Delete?")) { 
            await fetch('/delete_chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({filename:file})}); 
            await loadSidebar(); 
        }
    };
    async function loadChat(file) {
        diffModeMsgs.clear();
        isComparing = false;
        await fetch('/load_chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({filename:file})});
        await loadHistory(); 
        const sidebar = document.getElementById('sidebar');
        if (sidebar) sidebar.classList.add('hidden');
    }

    safeBind('live-sum-enabled', 'change', (e) => {
        const bgSumOpts = document.getElementById('sum-options');
        const sumEnabled = document.getElementById('sum-enabled')?.checked;
        if (bgSumOpts) {
            bgSumOpts.style.display = (e.target.checked || sumEnabled) ? 'block' : 'none';
        }
    });

    const openSettings = async () => {
        try {
            const [res, mRes] = await Promise.all([
                fetch('/get_settings'),
                fetch('/venice_models')
            ]);
            const d = await res.json();
            const mData = await mRes.json();

            const setCheck = (id, val) => { const el = document.getElementById(id); if (el) el.checked = val; };
            const setVal = (id, val) => { const el = document.getElementById(id); if (el) el.value = val; };

            const populateAllSyncSelectors = (currentSettings) => {
                const mappings = {
                    'set-model-chat': currentSettings.venice.model,
                    'ga-wfm': currentSettings.wfm.model,
                    'ga-refiner': currentSettings.refiner.model,
                    'ga-img': currentSettings.venice_img.model,
                    'ga-visual': currentSettings.venice_img.visual_scan_model,
                    'ga-sum': currentSettings.summarizer.model,
                    'ga-cons': currentSettings.summarizer.consolidation_model,
                    'ga-lore': currentSettings.rag.extraction_model || "venice-uncensored",
                    'ga-eval': currentSettings.evaluator_model || "venice-uncensored"
                };

                for (const [id, val] of Object.entries(mappings)) {
                    const select = document.getElementById(id);
                    if (select) {
                        select.dataset.currentValue = val;
                        populateFilteredSelect(select);
                    }
                }
            };

            populateAllSyncSelectors(d);

            const updateMiniModelInfo = (selectId, infoId) => {
                const select = document.getElementById(selectId);
                const infoDiv = document.getElementById(infoId);
                if (!select || !infoDiv || !availableModels) return;

                const modelId = select.value;
                let selectedModel = null;
                for (const group in availableModels) {
                    const found = availableModels[group].find(m => m.id === modelId);
                    if (found) { selectedModel = found; break; }
                }

                if (selectedModel) {
                    infoDiv.textContent = `CTX: ${selectedModel.context} | ${selectedModel.pricing}`;
                }
            };
            setCheck('set-venice-system', d.venice.include_venice_system_prompt ?? true);
            setCheck('set-vision-high-res', d.venice.vision_high_res ?? true);
            setVal('set-temp-chat', d.venice.temperature);
            setVal('set-tokens-chat', d.venice.max_tokens);
            setVal('set-reasoning-effort', d.venice.reasoning_effort || "medium");
            setVal('set-freq-chat', d.venice.frequency_penalty);
            setVal('set-pres-chat', d.venice.presence_penalty);

            // Populate Model Info Panel
            const modelSelect = document.getElementById('set-model-chat');
            const modelInfoPanel = document.getElementById('model-info-panel');

            const updateModelInfo = (modelId) => {
                let selectedModel = null;
                for (const group in mData) {
                    const found = mData[group].find(m => m.id === modelId);
                    if (found) { selectedModel = found; break; }
                }

                if (selectedModel && modelInfoPanel) {
                    modelInfoPanel.style.display = 'block';
                    document.getElementById('mi-name').textContent = selectedModel.name;
                    document.getElementById('mi-pricing').textContent = `In/Out: ${selectedModel.pricing}`;
                    document.getElementById('mi-context').textContent = `${selectedModel.context} Context`;
                    document.getElementById('mi-traits').textContent = selectedModel.traits;
                    document.getElementById('mi-desc').textContent = selectedModel.description;

                    const tagsContainer = document.getElementById('mi-tags');
                    tagsContainer.innerHTML = '';
                    selectedModel.tags.forEach(tag => {
                        const tSpan = document.createElement('span');
                        tSpan.style.cssText = "background:rgba(255,255,255,0.05); border:1px solid #444; padding:2px 6px; border-radius:4px; font-size:0.8em; color:#aaa;";
                        tSpan.textContent = tag;
                        tagsContainer.appendChild(tSpan);
                    });
                } else if (modelInfoPanel) {
                    modelInfoPanel.style.display = 'none';
                }

                const reasoningContainer = document.getElementById('reasoning-effort-container');
                if (reasoningContainer) {
                    const isReasoning = modelId.startsWith('openai-gpt-') || 
                                       modelId.startsWith('claude-') || 
                                       modelId.startsWith('google-gemini-') ||
                                       modelId.includes('deepseek-') || 
                                       modelId.includes('kimi-k2') || 
                                       modelId.includes('minimax-m2') ||
                                       modelId.includes('glm-4.7') ||
                                       modelId.includes('glm-5') ||
                                       modelId.includes('qwen3-5-35b') ||
                                       modelId.includes('qwen3-vl-235b') ||
                                       modelId.includes('trinity-large-thinking');
                    reasoningContainer.style.display = isReasoning ? 'block' : 'none';
                }
            };

            // Sync other Global Assignment selectors to their counterparts
            const syncGA = (gaId, targetId) => {
                const ga = document.getElementById(gaId);
                const target = document.getElementById(targetId);
                if (ga && target) {
                    ga.addEventListener('change', (e) => {
                        target.value = e.target.value;
                    });
                }
            };

            syncGA('ga-wfm', 'set-model-wfm');
            syncGA('ga-refiner', 'set-model-refiner');
            syncGA('ga-img', 'set-model-img');
            syncGA('ga-visual', 'set-model-visual-scan');
            syncGA('ga-sum', 'sum-model');
            syncGA('ga-cons', 'sum-cons-model');

            if (modelSelect) {
                modelSelect.addEventListener('change', (e) => updateModelInfo(e.target.value));
                if (modelSelect.value) updateModelInfo(modelSelect.value);
            }
            
            document.querySelectorAll('.model-selector-sync').forEach(select => {
                if (select.id.startsWith('ga-')) {
                    select.addEventListener('change', () => updateMiniModelInfo(select.id, 'mi-' + select.id.split('-')[1]));
                }
            });

            setVal('set-temp-wfm', d.wfm.temperature || 0.8);
            setVal('set-depth-wfm', d.wfm.context_depth || 10);

            setVal('set-temp-refiner', d.refiner?.temperature || 0.3);

            setVal('set-art-style', d.image_gen.art_style || "None");
            setVal('set-negative-styles', d.image_gen.negative_styles || "");
            setVal('set-img-depth', d.venice_img.context_depth || 3);

            const fontSize = d.interface?.font_size || 16;
            const bgColor = d.interface?.bg_color || "#121212";
            setVal('set-font-size', fontSize);
            setVal('set-bg-color', bgColor);
            applyInterfaceSettings(fontSize, bgColor);

            setVal('set-ar', d.image_gen.aspect_ratio || "3:4");
            const resTier = d.image_gen.res_tier || "standard";
            const resRadio = document.querySelector(`input[name="res-tier"][value="${resTier}"]`);
            if (resRadio) resRadio.checked = true;

            guidedModeEnabled = d.image_gen.guided_mode || false;
            setCheck('set-guided', guidedModeEnabled);

            setCheck('sum-enabled', d.summarizer.enabled);
            setCheck('live-sum-enabled', d.summarizer.live_summary_enabled || false);

            setCheck('set-auto-memory', d.venice.auto_memory_enabled ?? false);
            setVal('sum-threshold', d.summarizer.trigger_threshold_turns);
            setVal('sum-batch', d.summarizer.batch_size);
            setVal('sum-keep', d.summarizer.recent_turns_to_keep);
            setVal('sum-prompt', d.summarizer.system_prompt);
            setVal('sum-cons-prompt', d.summarizer.consolidation_prompt || "Summarize.");

            const sumOpts = document.getElementById('sum-options');
            if (sumOpts) {
                sumOpts.style.display = (d.summarizer.enabled || d.summarizer.live_summary_enabled) ? 'block' : 'none';
                const bgFields1 = document.getElementById('bg-sum-fields');
                const bgFields2 = document.getElementById('bg-sum-prompt-field');
                const displayBg = d.summarizer.enabled ? 'block' : 'none';
                if (bgFields1) bgFields1.style.display = displayBg;
                if (bgFields2) bgFields2.style.display = displayBg;
            }

            setCheck('rag-enabled', d.rag.enabled);
            setVal('rag-k', d.rag.k);
            setVal('rag-max-chars', d.rag.max_chars);
            setVal('rag-min-chars', d.rag.min_chars);

            const ragOpts = document.getElementById('rag-options');
            if (ragOpts) ragOpts.style.display = d.rag.enabled ? 'block' : 'none';

            setVal('set-tts-model', d.tts.model || "tts-kokoro");

            const mistralIdInput = document.getElementById('mistral-voice-id');
            if (d.tts.model.includes('voxtral')) {
                if (mistralIdInput) mistralIdInput.value = d.tts.voice || "";
            } else {
                setVal('set-tts-voice', d.tts.voice || "af_sky");
            }
            setVal('set-tts-speed', d.tts.speed || 1.0);
            const speedDisp = document.getElementById('tts-speed-val');
            if(speedDisp) speedDisp.textContent = (d.tts.speed || 1.0) + "x";

            const ttsOpts = document.getElementById('tts-options');
            if (ttsOpts) ttsOpts.style.display = d.tts.enabled ? 'block' : 'none';

            // Mistral UI Priority Feedback
            const mistralIdInput_fb = document.getElementById('mistral-voice-id');
            const cloneStatus = document.getElementById('tts-clone-status');
            const clearCloneBtn = document.getElementById('clear-tts-clone-btn');

            if (ttsRefAudioBase64) {
                if (cloneStatus) cloneStatus.style.display = 'block';
                if (clearCloneBtn) clearCloneBtn.style.display = 'block';
            }

            if (modal) modal.style.display = 'block';
            checkScanStatus();
        } catch(e) {
            console.error("Failed to open settings:", e);
        }
    }

    function calcPixels(ar, resTier) {
        let w = 1024, h = 1024;

        // Base Aspect Ratio logic (Standard ~1MP)
        if (ar === "1:1") { w = 1024; h = 1024; }
        else if (ar === "4:3") { w = 1152; h = 864; }
        else if (ar === "3:4") { w = 864; h = 1152; }
        else if (ar === "16:9") { w = 1216; h = 704; }
        else if (ar === "9:16") { w = 704; h = 1216; }
        else if (ar === "2:3") { w = 832; h = 1248; }
        else if (ar === "3:2") { w = 1248; h = 832; }

        if (resTier === "hd") {
            // HD Tier: Scale linearly by ~1.4x (Approx 2MP)
            w = Math.floor((w * 1.414) / 32) * 32;
            h = Math.floor((h * 1.414) / 32) * 32;
        } else if (resTier === "2k") {
            // 2K Tier: Longest side is exactly 2048 (Approx 4MP)
            const ratio = w / h;
            if (w >= h) {
                w = 2048;
                h = Math.floor((2048 / ratio) / 32) * 32;
            } else {
                h = 2048;
                w = Math.floor((2048 * ratio) / 32) * 32;
            }
        }

        // Final safety bounds for Z-Image Turbo (Max 4MP / 2048px)
        w = Math.min(Math.max(w, 256), 2048);
        h = Math.min(Math.max(h, 256), 2048);

        return { w, h };
    }

    async function runAnalytics() {
        console.log("Analyze running...");
        const startNum = parseInt(document.getElementById('ana-start')?.value);
        const endNum = parseInt(document.getElementById('ana-end')?.value);
        const role = document.getElementById('ana-role')?.value || 'both';

        if (isNaN(startNum) || isNaN(endNum) || startNum > endNum) {
            alert("Invalid range."); return;
        }

        let textMsgCounter = 0;
        let actualStartIdx = -1;
        let actualEndIdx = -1;

        for (let i = 0; i < chatHistory.length; i++) {
            const m = chatHistory[i];
            if (m.role === 'system' || (typeof m.content === 'string' && m.content.startsWith('__IMG_JSON__'))) continue;
            textMsgCounter++;
            if (textMsgCounter === startNum) actualStartIdx = i;
            if (textMsgCounter === endNum) actualEndIdx = i;
        }

        if (actualStartIdx === -1) actualStartIdx = 0;
        if (actualEndIdx === -1) actualEndIdx = chatHistory.length - 1;

        const resultsDiv = document.getElementById('analytics-results');
        const loadingDiv = document.getElementById('analytics-loading');

        if(resultsDiv) resultsDiv.style.display = 'none';
        if(loadingDiv) loadingDiv.style.display = 'block';

        try {
            const res = await fetch('/analyze_chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ start: actualStartIdx, end: actualEndIdx, role: role })
            });
            const d = await res.json();

            if(loadingDiv) loadingDiv.style.display = 'none';

            if (d.success) {
                if(resultsDiv) resultsDiv.style.display = 'block';

                const setStat = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };

                setStat('stat-msgs', d.stats.msg_count);
                setStat('stat-words', d.stats.total_words.toLocaleString());
                setStat('stat-vocab', d.stats.unique_words.toLocaleString());
                setStat('stat-richness', d.stats.lexical_richness + '%');
                setStat('stat-avg', d.stats.avg_words_per_msg);

                const renderList = (elementId, items) => {
                    const container = document.getElementById(elementId);
                    if(!container) return;
                    container.innerHTML = '';
                    if (!items || items.length === 0) {
                        container.innerHTML = '<div style="color:#666; font-style:italic;">No data</div>';
                        return;
                    }
                    const maxCount = items[0][1];
                    items.forEach(item => {
                        const word = item[0];
                        const count = item[1];
                        const pct = (count / maxCount) * 100;

                        const row = document.createElement('div');
                        row.className = 'ana-row';
                        row.innerHTML = `
                            <div style="display:flex; justify-content:space-between; margin-bottom:2px; position:relative; z-index:2;">
                                <span style="font-weight:bold; color:#ddd;">${word}</span>
                                <span style="color:var(--text-dim); font-size:0.9em;">${count}</span>
                            </div>
                            <div class="ana-bar" style="width:${pct}%;"></div>
                        `;
                        row.title = "Click to append to Banned Phrases";
                        row.style.cursor = "pointer";
                        row.onclick = () => {
                            if(confirm(`Add "${word}" to your Banned Phrases list?`)) {
                                const ta = document.getElementById('set-banned-phrases');
                                if (ta) {
                                    const curr = ta.value.trim();
                                    ta.value = curr ? curr + "\n" + word : word;
                                    alert(`"${word}" appended! Don't forget to click 'Save All Changes' in the Settings menu later.`);
                                }
                            }
                        };
                        container.appendChild(row);
                    });
                };

                renderList('ana-1gram', d.unigrams);
                renderList('ana-2gram', d.bigrams);
                renderList('ana-3gram', d.trigrams);
                renderList('ana-4gram', d.quadgrams);
                renderList('ana-5gram', d.pentagrams);

            } else {
                alert("Analytics Error: " + (d.error || "Unknown error"));
            }
        } catch(e) {
            console.error("Analytics request failed:", e);
            if(loadingDiv) loadingDiv.style.display = 'none';
            alert("Failed to run analytics. Check console for details.");
        }
    }


    // --- SAFE EVENT BINDINGS ---

    if (userInput) {
        userInput.addEventListener('keydown', (e) => { 
            if(e.key === 'Enter' && !e.shiftKey) { 
                e.preventDefault();
                const start = userInput.selectionStart; const end = userInput.selectionEnd; const val = userInput.value;
                userInput.value = val.substring(0, start) + "\n" + val.substring(end);
                userInput.selectionStart = userInput.selectionEnd = start + 1;
            } 
        });
    }

    safeBind('dismiss-compare-btn', 'click', async () => {
        if (!confirm("Discard the backup summaries? This will disable Comparison Mode for this chat session.")) return;
        try {
            const res = await fetch('/clear_backups', { method: 'POST' });
            if (res.ok) {
                await loadHistory();
            }
        } catch (e) {
            alert("Failed to clear backups.");
        }
    });

    safeBind('sendBtn', 'click', sendMessage);
    safeBind('genImgBtn', 'click', handleImageBtn);
    safeBind('toggleGuidedBtn', 'click', () => {
        if (!guidedModal) return;
        guidedModal.style.display = (guidedModal.style.display === 'none') ? 'block' : 'none';
        if (guidedModal.style.display === 'block' && guidedInput) guidedInput.focus();
    });
    safeBind('guided-submit', 'click', () => {
        const debugMode = document.getElementById('guided-debug-mode')?.checked || false;
        triggerImageGen(guidedInput?.value, debugMode);
    });
    safeBind('guided-cancel', 'click', () => { 
        const modal = document.getElementById('guided-modal');
        if(modal) modal.style.display='none'; 
    });
    safeBind('guided-cancel-x', 'click', () => { 
        const modal = document.getElementById('guided-modal');
        if(modal) modal.style.display='none'; 
    });

    function showDebugModal(debugData) {
        const modal = document.getElementById('debug-modal');
        const payloadEl = document.getElementById('debug-payload');
        const responseEl = document.getElementById('debug-response');
        if (modal && payloadEl && responseEl) {
            // Check if generation or request failed
            payloadEl.textContent = JSON.stringify(debugData, null, 2);
            responseEl.textContent = "See detailed object above for full API history.";
            modal.style.display = 'block';
        }
    }

    safeBind('close-debug-x', 'click', () => {
        const modal = document.getElementById('debug-modal');
        if (modal) modal.style.display = 'none';
    });
    safeBind('menuBtn', 'click', openSettings);

    safeBind('mobile-menu-btn', 'click', () => { 
        loadSidebar(); 
        const sb = document.getElementById('sidebar');
        if (sb) {
            // Give a tiny delay so the click outside event doesn't immediately fire
            setTimeout(() => sb.classList.remove('hidden'), 10);
        }
    });

    // Close sidebar by clicking outside
    const mainEl = document.getElementById('main');
    if (mainEl) {
        mainEl.addEventListener('click', (e) => {
            const sb = document.getElementById('sidebar');
            if (sb && !sb.classList.contains('hidden')) {
                // Ensure we didn't click the menu button itself
                if (!e.target.closest('#mobile-menu-btn')) {
                    sb.classList.add('hidden');
                }
            }
        });
    }

    safeBind('close-sidebar', 'click', () => {
        const sb = document.getElementById('sidebar');
        if (sb) sb.classList.add('hidden');
    });

    safeBind('new-chat-btn', 'click', async () => { 
        await fetch('/new_chat', {method:'POST'}); 
        loadHistory(); 
        const sb = document.getElementById('sidebar');
        if (sb) sb.classList.add('hidden'); 
    });

    // ARENA MODE BINDINGS
    window.arenaBranchIndex = null;

    safeBind('new-arena-btn', 'click', async () => {
        window.arenaBranchIndex = null;
        const mData = availableModels || await fetch('/venice_models').then(r => r.json());
        const container = document.getElementById('arena-model-checkboxes');
        container.innerHTML = '';
        
        for (const group in mData) {
            const groupDiv = document.createElement('div');
            groupDiv.style.cssText = "margin-top:10px; font-weight:bold; color:var(--purple); font-size:0.9em;";
            groupDiv.textContent = group;
            container.appendChild(groupDiv);
            
            mData[group].forEach(m => {
                const lbl = document.createElement('label');
                lbl.style.cssText = "display:flex; align-items:center; gap:8px; padding:5px; margin-left:10px; color:#ccc; font-weight:normal; cursor:pointer;";
                lbl.innerHTML = `<input type="checkbox" value="${m.id}" data-name="${m.name}"> ${m.name}`;
                container.appendChild(lbl);
            });
        }
        document.getElementById('arena-setup-modal').style.display = 'block';
        const sb = document.getElementById('sidebar');
        if (sb) sb.classList.add('hidden');
    });

    window.handleBranchToArena = async (idx) => {
        window.arenaBranchIndex = idx;
        if (!availableModels) {
            const r = await fetch('/venice_models');
            availableModels = await r.json();
        }
        const container = document.getElementById('arena-model-checkboxes');
        if (!container) return;
        container.innerHTML = '';
        
        for (const group in availableModels) {
            const groupDiv = document.createElement('div');
            groupDiv.style.cssText = "margin-top:10px; font-weight:bold; color:var(--purple); font-size:0.9em;";
            groupDiv.textContent = group;
            container.appendChild(groupDiv);
            
            availableModels[group].forEach(m => {
                const lbl = document.createElement('label');
                lbl.style.cssText = "display:flex; align-items:center; gap:8px; padding:5px; margin-left:10px; color:#ccc; font-weight:normal; cursor:pointer;";
                lbl.innerHTML = `<input type="checkbox" value="${m.id}" data-name="${m.name}"> ${m.name}`;
                container.appendChild(lbl);
            });
        }
        document.getElementById('arena-setup-modal').style.display = 'block';
    };

    safeBind('close-arena-setup-x', 'click', () => {
        document.getElementById('arena-setup-modal').style.display = 'none';
        window.arenaBranchIndex = null;
    });

    safeBind('start-arena-btn', 'click', async () => {
        const cbs = document.querySelectorAll('#arena-model-checkboxes input:checked');
        if (cbs.length < 2) { alert("Select at least 2 models."); return; }
        
        const selected = Array.from(cbs).map(c => ({id: c.value, name: c.dataset.name}));
        const btn = document.getElementById('start-arena-btn');
        btn.textContent = "Starting..."; btn.disabled = true;
        
        try {
            const endpoint = window.arenaBranchIndex !== null ? '/branch_to_arena' : '/create_arena';
            const payload = {models: selected};
            if (window.arenaBranchIndex !== null) payload.index = window.arenaBranchIndex;

            const res = await fetch(endpoint, {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
            const d = await res.json();
            if (d.success) {
                document.getElementById('arena-setup-modal').style.display = 'none';
                window.arenaBranchIndex = null;
                await loadSidebar();
                await loadChat(d.filename);
            } else {
                alert("Arena Error: " + (d.error || "Unknown error"));
            }
        } catch(e) { 
            console.error(e);
            alert("Failed to start Arena."); 
        }
        btn.textContent = "Start Arena"; btn.disabled = false;
    });

    safeBind('arena-prev-btn', 'click', () => {
        if (!window.arenaData) return;
        if (window.activeArenaTab === 'evaluator') {
            window.activeArenaTab = window.arenaData.models[window.arenaData.models.length-1].id;
        } else {
            const idx = window.arenaData.models.findIndex(m => m.id === window.activeArenaTab);
            if (idx > 0) window.activeArenaTab = window.arenaData.models[idx-1].id;
            else window.activeArenaTab = 'evaluator';
        }
        renderArenaChat();
    });

    safeBind('arena-next-btn', 'click', () => {
        if (!window.arenaData) return;
        if (window.activeArenaTab === 'evaluator') {
            window.activeArenaTab = window.arenaData.models[0].id;
        } else {
            const idx = window.arenaData.models.findIndex(m => m.id === window.activeArenaTab);
            if (idx < window.arenaData.models.length - 1) window.activeArenaTab = window.arenaData.models[idx+1].id;
            else window.activeArenaTab = 'evaluator';
        }
        renderArenaChat();
    });

    safeBind('arena-eval-btn', 'click', () => {
        window.activeArenaTab = 'evaluator';
        renderArenaChat();
    });

    function isGeneratingArena() {
        // Simple check to see if we're in arena mode and generation is active
        return window.isArenaMode && isGenerating;
    }

    function renderArenaChat() {
        const chatbox = document.getElementById('chatbox');
        if (!chatbox) return;
        chatbox.innerHTML = '';

        // [INJECTION] Check for Pipeline
        if (window.arenaData && window.arenaData.chat_type === 'pipeline') {
            document.getElementById('pipeline-workspace').style.display = 'flex';
            currentPipelinePhase = window.arenaData.pipeline_phase || 'architect';
            document.getElementById('pipeline-phase-badge').textContent = `PHASE: ${currentPipelinePhase.toUpperCase()}`;
            
            if (currentPipelinePhase === 'scribe') {
                document.getElementById('pipeline-lock-btn').textContent = '← Back to Architect';
                document.getElementById('pipeline-lock-btn').style.background = '#444';
            } else {
                document.getElementById('pipeline-lock-btn').textContent = 'Lock Blueprint & Move to Scribe ▶';
                document.getElementById('pipeline-lock-btn').style.background = 'var(--purple)';
            }
            
            document.getElementById('blueprint-editor').value = window.arenaData.blueprint || '';
            
            // Render history into pipeline chatbox instead of main chatbox
            const pbox = document.getElementById('pipeline-chatbox');
            pbox.innerHTML = '';
            window.arenaData.history.forEach((msg, idx) => {
                if (msg.role !== 'system') {
                    let html = `<div class="message ${msg.role}-msg"><div class="msg-content">${marked.parse(msg.content)}</div></div>`;
                    pbox.innerHTML += html;
                }
            });
            pbox.scrollTop = pbox.scrollHeight;
            return; // Exit normal chatbox rendering
        } else {
            const pWork = document.getElementById('pipeline-workspace');
            if (pWork) pWork.style.display = 'none';
        }

        
        const isEval = window.activeArenaTab === 'evaluator';
        const aRow = document.getElementById('arena-controls-row');
        if (aRow) aRow.style.display = isEval ? 'none' : 'flex';
        const eRow = document.getElementById('evaluator-controls-row');
        if (eRow) eRow.style.display = isEval ? 'flex' : 'none';
        
        const label = document.getElementById('arena-current-tab');
        if (isEval) {
            label.textContent = "âš–ï¸ Evaluator Mode";
        } else {
            const m = window.arenaData.models.find(x => x.id === window.activeArenaTab);
            label.textContent = m ? m.name : window.activeArenaTab;
        }
        
        const msgs = isEval ? window.arenaData.evaluator : window.arenaData.threads[window.activeArenaTab];
        
        msgs.forEach((msg, idx) => {
            if (msg.role === 'system') return;
            
            const div = document.createElement('div');
            div.className = `message ${msg.role}`;
            div.id = `msg-arena-${idx}`;

            if (msg.timestamp) {
                const ts = document.createElement('div');
                ts.className = 'msg-timestamp';
                ts.textContent = new Date(msg.timestamp).toLocaleTimeString([], { hour12: true, hour: 'numeric', minute: '2-digit', second: '2-digit' });
                div.appendChild(ts);
            }

            // Reasoning Block
            if (msg.reasoning) {
                const rBlock = document.createElement('div');
                rBlock.className = 'reasoning-block';
                // If it's the last message and it's generating, keep it expanded
                const isGeneratingMsg = (idx === msgs.length - 1 && msg.role === 'assistant');
                const displayStyle = isGeneratingMsg ? 'block' : 'none';
                const toggleChar = isGeneratingMsg ? '▲' : '▼';
                
                rBlock.innerHTML = `
                    <div class="reasoning-header">
                        <span><i style="margin-right:5px;">💭</i> Thought Process</span>
                        <span class="toggle-icon">${toggleChar}</span>
                    </div>
                    <div class="reasoning-content" style="display:${displayStyle};"></div>
                `;
                rBlock.querySelector('.reasoning-content').textContent = msg.reasoning;
                rBlock.querySelector('.reasoning-header').onclick = () => {
                    const content = rBlock.querySelector('.reasoning-content');
                    const icon = rBlock.querySelector('.toggle-icon');
                    const isHidden = content.style.display === 'none';
                    content.style.display = isHidden ? 'block' : 'none';
                    icon.textContent = isHidden ? '▲' : '▼';
                };
                div.appendChild(rBlock);
            }

            // Blueprint Tool Calls Block (persisted on message)
            if (msg.blueprint_tool_calls) {
                const tcBlock = document.createElement('div');
                tcBlock.className = 'tool-calls-block';
                tcBlock.style.cssText = 'margin-top:8px; border:1px solid #2d4a2d; border-radius:6px; overflow:hidden; font-size:0.85em;';
                tcBlock.innerHTML = `
                    <div class="tool-calls-header" style="display:flex; justify-content:space-between; align-items:center; padding:6px 12px; background:#1a2e1a; cursor:pointer; user-select:none;">
                        <span><i style="margin-right:5px;">🔧</i>Blueprint Tool Calls</span>
                        <span class="toggle-icon-tc">▼</span>
                    </div>
                    <div class="tool-calls-content" style="display:none; padding:12px; background:#111; white-space:pre-wrap; font-family:monospace; color:#a3e635; overflow-x:auto;"></div>
                `;
                tcBlock.querySelector('.tool-calls-content').textContent = msg.blueprint_tool_calls;
                tcBlock.querySelector('.tool-calls-header').onclick = () => {
                    const c = tcBlock.querySelector('.tool-calls-content');
                    const ic = tcBlock.querySelector('.toggle-icon-tc');
                    const isH = c.style.display === 'none';
                    c.style.display = isH ? 'block' : 'none';
                    ic.textContent = isH ? '▲' : '▼';
                };
                div.appendChild(tcBlock);
            }

            const contentDiv = document.createElement('div');
            contentDiv.className = 'msg-content';
            contentDiv.style.cssText = "white-space:pre-wrap; font-family:inherit;";
            
            let textContent = msg.content;
            if (!textContent && msg.role === 'assistant') {
                if (msg.reasoning) {
                    textContent = '';
                } else {
                    // Check if this is the absolute last message in a currently generating arena
                    const isGenerating = (idx === msgs.length - 1 && isGeneratingArena()); 
                    textContent = isGenerating ? '<i>Thinking...</i>' : '';
                }
            }
            contentDiv.innerHTML = typeof marked !== 'undefined' ? marked.parse(textContent || "") : (textContent || "");
            div.appendChild(contentDiv);

            // Options/Stats Panel
            const optionsBtn = document.createElement('div');
            optionsBtn.className = 'options-btn';
            optionsBtn.innerHTML = '&#8942;'; // Vertical Ellipsis

            const toolsPanel = document.createElement('div');
            toolsPanel.className = 'tools-panel';

            const stats = document.createElement('div');
            stats.className = 'stats-block';
            let statHtml = `Arena Msg`;
            if (msg.model) statHtml += `<br>Model: <span style="color:var(--primary);">${msg.model}</span>`;
            if (msg.usage) {
                const u = msg.usage;
                statHtml += `<br>Usage: In:${u.prompt_tokens} Out:${u.completion_tokens}`;
                const pricing = getModelPricing(msg.model);
                if (pricing) {
                    const totalCost = (u.prompt_tokens / 1000000 * pricing.input) + (u.completion_tokens / 1000000 * pricing.output);
                    statHtml += `<br>Est. Cost: <span style="color:var(--accent);">${totalCost.toFixed(5)}</span>`;
                }
            }
            stats.innerHTML = statHtml;
            toolsPanel.appendChild(stats);

            optionsBtn.onclick = () => { toolsPanel.style.display = (toolsPanel.style.display === 'block') ? 'none' : 'block'; };
            div.appendChild(optionsBtn);
            div.appendChild(toolsPanel);

            chatbox.appendChild(div);
        });
        chatbox.scrollTop = chatbox.scrollHeight;
    }

    // --- ARENA STREAMING OPTIMIZATION ---
    let arenaUpdateQueue = [];
    let isAnimationFrameRequested = false;

    function processArenaUpdateQueue() {
        while (arenaUpdateQueue.length > 0) {
            const { mId, update } = arenaUpdateQueue.shift();
            
            // Only update the DOM if the user is looking at this model's tab
            if (window.activeArenaTab !== mId) continue;

            const thread = window.arenaData.threads[mId] || window.arenaData.evaluator;
            const lastMsgIdx = thread.length - 1;
            const msgDiv = document.getElementById(`msg-arena-${lastMsgIdx}`);
            
            if (!msgDiv) continue;

            if (update.content) {
                const contentDiv = msgDiv.querySelector('.msg-content');
                if (contentDiv) {
                    // Remove 'Thinking...' placeholder if it's there
                    if (contentDiv.innerHTML === '<i>Thinking...</i>') {
                        contentDiv.innerHTML = '';
                    }
                    contentDiv.innerHTML = typeof marked !== 'undefined' ? marked.parse(thread[lastMsgIdx].content) : thread[lastMsgIdx].content;
                }
            }
            if (update.reasoning) {
                const rContent = msgDiv.querySelector('.reasoning-content');
                if (rContent) {
                    rContent.textContent = thread[lastMsgIdx].reasoning;
                    rContent.style.display = 'block';
                    const icon = msgDiv.querySelector('.toggle-icon');
                    if (icon) icon.textContent = '▲';
                }
                // Also clear thinking placeholder if content hasn't arrived yet but reasoning has
                const contentDiv = msgDiv.querySelector('.msg-content');
                if (contentDiv && contentDiv.innerHTML === '<i>Thinking...</i>') {
                    contentDiv.innerHTML = '';
                }
            }
            if (update.usage) {
                const stats = msgDiv.querySelector('.stats-block');
                if (stats) {
                    const u = update.usage;
                    let statHtml = `Arena Msg<br>Model: <span style="color:var(--primary);">${thread[lastMsgIdx].model || mId}</span>`;
                    statHtml += `<br>Usage: In:${u.prompt_tokens} Out:${u.completion_tokens}`;
                    stats.innerHTML = statHtml;
                }
            }
        }

        // Maintain sticky scroll
        const isAtBottom = chatbox.scrollHeight - chatbox.scrollTop <= chatbox.clientHeight + 150;
        if (isAtBottom) chatbox.scrollTop = chatbox.scrollHeight;
        
        isAnimationFrameRequested = false;
    }

    function queueArenaUIUpdate(mId, update) {
        arenaUpdateQueue.push({ mId, update });
        if (!isAnimationFrameRequested) {
            isAnimationFrameRequested = true;
            requestAnimationFrame(processArenaUpdateQueue);
        }
    }

    async function sendArenaMessage(txt) {
        const isEval = window.activeArenaTab === 'evaluator';
        const sendTime = new Date().toISOString();
        
        if (isEval) {
            const evalModel = document.getElementById('arena-eval-model-select').value;
            window.arenaData.evaluator.push({role: 'user', content: txt, timestamp: sendTime});
            window.arenaData.evaluator.push({role: 'assistant', content: '', model: evalModel, timestamp: new Date().toISOString()});
            renderArenaChat();
            
            const hideNames = document.getElementById('eval-hide-names').checked;
            
            try {
                const res = await fetch('/arena_eval', { 
                    method:'POST', 
                    headers:{'Content-Type':'application/json'}, 
                    body:JSON.stringify({message: txt, hide_names: hideNames, model: evalModel}) 
                });
                const reader = res.body.getReader();
                const dec = new TextDecoder();
                let buffer = "";
                
                while(true){
                    const {done, value} = await reader.read();
                    if(done) break;
                    
                    buffer += dec.decode(value, {stream: true});
                    let lines = buffer.split('\n\n');
                    buffer = lines.pop();

                    for (const l of lines) {
                        if(l.startsWith('data: ')) {
                            try { 
                                const d = JSON.parse(l.substring(6)); 
                                const thread = window.arenaData.evaluator;
                                const lastMsg = thread[thread.length - 1];

                                if (d.content) lastMsg.content += d.content;
                                if (d.reasoning) lastMsg.reasoning += d.reasoning;
                                if (d.usage) lastMsg.usage = d.usage;
                                
                                queueArenaUIUpdate('evaluator', d);
                            } catch(e){}
                        }
                    }
                }
            } catch(e) { console.error("Arena Eval Error:", e); }
            await loadHistory();
        } else {
            const msgObj = {role: 'user', content: txt, timestamp: sendTime};
            const target = document.getElementById('arena-target-select').value;
            const targets = target === 'all' ? window.arenaData.models.map(m=>m.id) : [window.activeArenaTab];
            
            targets.forEach(m => {
                window.arenaData.threads[m].push(msgObj);
                window.arenaData.threads[m].push({role: 'assistant', content: '', model: m, timestamp: new Date().toISOString()});
            });
            renderArenaChat();
            
            try {
                const reqBody = {message: txt, target: target === 'all' ? 'all' : window.activeArenaTab};
                const res = await fetch('/arena_chat', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(reqBody) });
                const reader = res.body.getReader();
                const dec = new TextDecoder();
                let buffer = "";
                
                while(true){
                    const {done, value} = await reader.read();
                    if(done) break;
                    
                    buffer += dec.decode(value, {stream: true});
                    let lines = buffer.split('\n\n');
                    buffer = lines.pop();

                    for (const l of lines) {
                        if(l.startsWith('data: ')) {
                            try { 
                                const d = JSON.parse(l.substring(6)); 
                                const mId = Object.keys(d)[0];
                                const thread = window.arenaData.threads[mId];
                                const update = d[mId];
                                
                                const lastMsg = thread[thread.length - 1];
                                if (update.content) lastMsg.content += update.content;
                                if (update.reasoning) lastMsg.reasoning += update.reasoning;
                                if (update.usage) lastMsg.usage = update.usage;
                                
                                queueArenaUIUpdate(mId, update);
                            } catch(e){}
                        }
                    }
                }
            } catch(e) { console.error("Arena Chat Error:", e); }
            await loadHistory();
        }
    }

    // AUDIT MODE BINDINGS
    safeBind('open-audit-btn', 'click', async () => {
        const sbRes = await fetch('/sidebar_data');
        const sbData = await sbRes.json();
        const activeFile = sbData.active_chat;
        
        if (!activeFile || activeFile.endsWith('.audit.json')) {
            alert("Please select a standard chat first to create an audit for it.");
            return;
        }
        
        try {
            const res = await fetch('/open_audit', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ filename: activeFile })
            });
            const d = await res.json();
            if (d.success) {
                await loadSidebar();
                await loadChat(d.filename);
            }
        } catch (e) {
            alert("Failed to open audit chat.");
        }
    });

    async function fetchParentContext() {
        try {
            const res = await fetch('/get_parent_context');
            const d = await res.json();
            if (d.success) parentSummariesCache = d.summaries || [];
        } catch(e) { console.error("Failed to load parent context", e); }
    }

    function updateAuditTopBar() {
        const countSpan = document.getElementById('audit-batch-count');
        if (countSpan) countSpan.textContent = auditContextSelection.batches.length;
        
        // Sync the auto-include raw checkbox if drawer is built
        const rawCb = document.getElementById('audit-include-raw');
        if (rawCb) rawCb.checked = auditContextSelection.includeRaw !== false;
    }

    async function saveAuditContext() {
        updateAuditTopBar();
        try {
            await fetch('/save_audit_context', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(auditContextSelection)
            });
        } catch(e) { console.error("Failed to save audit context."); }
    }

    function renderAuditDrawer() {
        const listDiv = document.getElementById('audit-batch-list');
        if (!listDiv) return;
        listDiv.innerHTML = '';
        
        if (parentSummariesCache.length === 0) {
            listDiv.innerHTML = '<div style="color:#aaa; padding:10px;">No summaries exist in the original chat yet.</div>';
            return;
        }
        
        parentSummariesCache.forEach((batch, idx) => {
            const isSelected = auditContextSelection.batches.includes(idx);
            
            const card = document.createElement('div');
            card.className = 'audit-drawer-card' + (isSelected ? ' selected' : '');
            
            const headRow = document.createElement('div');
            headRow.style.cssText = "display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;";
            
            const titleWrap = document.createElement('div');
            titleWrap.style.cssText = "display:flex; align-items:center; gap:8px;";
            
            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.value = idx;
            cb.checked = isSelected;
            cb.style.cssText = "width:18px; height:18px; cursor:pointer;";
            cb.onchange = (e) => {
                if (e.target.checked) {
                    if (!auditContextSelection.batches.includes(idx)) auditContextSelection.batches.push(idx);
                } else {
                    auditContextSelection.batches = auditContextSelection.batches.filter(b => b !== idx);
                }
                card.classList.toggle('selected', e.target.checked);
                saveAuditContext();
            };
            
            const title = document.createElement('div');
            title.style.cssText = "font-weight:bold; color:var(--primary);";
            title.textContent = `Batch #${idx + 1} (Msgs ${batch.start_index} - ${batch.end_index})`;
            
            titleWrap.appendChild(cb);
            titleWrap.appendChild(title);
            
            const expandBtn = document.createElement('button');
            expandBtn.className = 'msg-btn';
            expandBtn.textContent = 'View Sources ▼';
            
            headRow.appendChild(titleWrap);
            headRow.appendChild(expandBtn);
            
            const preview = document.createElement('div');
            preview.style.cssText = "font-size:0.85em; color:#bbb; line-height:1.4;";
            preview.textContent = batch.content.replace(/<think>.*?<\/think>/gs, '').trim();
            
            const sourcesDiv = document.createElement('div');
            sourcesDiv.style.cssText = "display:none; margin-top:10px; padding-top:10px; border-top:1px solid #333; font-size:0.85em; max-height:200px; overflow-y:auto; background:#111; padding:8px; border-radius:6px;";
            // Note: In a full app, we would fetch the raw messages here. For now we explain it.
            sourcesDiv.innerHTML = `<i style="color:#888;">Associated raw messages (${batch.start_index} to ${batch.end_index}) are automatically linked when this batch is selected.</i>`;
            
            expandBtn.onclick = () => {
                const isHidden = sourcesDiv.style.display === 'none';
                sourcesDiv.style.display = isHidden ? 'block' : 'none';
                expandBtn.textContent = isHidden ? 'Hide Sources ▲' : 'View Sources ▼';
            };
            
            card.appendChild(headRow);
            card.appendChild(preview);
            card.appendChild(sourcesDiv);
            
            listDiv.appendChild(card);
        });
    }

    safeBind('audit-top-bar', 'click', async () => {
        const drawer = document.getElementById('audit-drawer');
        if (!drawer) return;
        
        if (drawer.style.display === 'none') {
            if (parentSummariesCache.length === 0) await fetchParentContext();
            renderAuditDrawer();
            drawer.style.display = 'flex';
        } else {
            drawer.style.display = 'none';
        }
    });

    safeBind('close-audit-drawer', 'click', (e) => {
        e.stopPropagation();
        const drawer = document.getElementById('audit-drawer');
        if (drawer) drawer.style.display = 'none';
    });

    safeBind('audit-select-all', 'click', () => {
        if (!parentSummariesCache) return;
        auditContextSelection.batches = parentSummariesCache.map((_, i) => i);
        saveAuditContext();
        renderAuditDrawer();
    });

    safeBind('audit-clear-all', 'click', () => {
        auditContextSelection.batches = [];
        saveAuditContext();
        renderAuditDrawer();
    });

    safeBind('audit-include-raw', 'change', (e) => {
        auditContextSelection.includeRaw = e.target.checked;
        saveAuditContext();
    });
    
    // Setup event delegation for dynamically created "Apply Fix" buttons in the chat
    if (chatbox) {
        chatbox.addEventListener('click', async (e) => {
            if (e.target.classList.contains('apply-audit-btn')) {
                const btn = e.target;
                const idx = btn.getAttribute('data-index');
                const newText = decodeURIComponent(btn.getAttribute('data-text'));
                
                if (!confirm(`Apply this rewrite to Summary Batch #${parseInt(idx) + 1} in the original chat?`)) return;
                
                const ogTxt = btn.textContent;
                btn.textContent = "Applying..."; btn.disabled = true;
                
                try {
                    const res = await fetch('/apply_audit_fix', {
                        method: 'POST', headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ index: idx, new_text: newText })
                    });
                    const d = await res.json();
                    if (d.success) {
                        btn.textContent = "✅ Fix Applied";
                        btn.style.background = "var(--accent)";
                    } else {
                        alert("Error applying fix: " + d.error);
                        btn.textContent = ogTxt; btn.disabled = false;
                    }
                } catch(err) {
                    alert("Network error applying fix.");
                    btn.textContent = ogTxt; btn.disabled = false;
                }
            }
        });
    }

    safeBind('expand-main-prompt', 'click', () => { 
        if(modal) modal.style.display='none'; 
        openLargeEditor(null, 'prompt'); 
    });

    safeBind('expand-visual-btn', 'click', () => { 
        if(modal) modal.style.display='none'; 
        openLargeEditor(null, 'visual'); 
    });

    safeBind('close-modal-x', 'click', () => { 
        if (modal) modal.style.display = 'none'; 
    });

    let scanStatusInterval = null;

    const updateScanUI = (job) => {
        const btn = document.getElementById('scan-visuals-btn');
        const statusDisplay = document.getElementById('scan-status-indicator');
        
        if (!statusDisplay) {
            // Create a small indicator if it doesn't exist
            const newIndicator = document.createElement('div');
            newIndicator.id = 'scan-status-indicator';
            newIndicator.style.cssText = "font-size:0.75em; padding:5px; margin-top:5px; border-radius:4px; text-align:center; display:none;";
            const btnParent = btn.parentNode;
            btnParent.insertBefore(newIndicator, btn.nextSibling);
        }

        const indicator = document.getElementById('scan-status-indicator');

        if (job.status === 'running') {
            btn.disabled = true;
            btn.textContent = "Scanning...";
            indicator.style.display = 'block';
            indicator.style.background = 'rgba(139, 92, 246, 0.2)';
            indicator.style.color = 'var(--purple)';
            indicator.innerHTML = `<span class="spinner" style="display:inline-block; vertical-align:middle; margin-right:5px;"></span> ${job.message}`;
            
            // If settings is open, ensure we keep polling
            if (!scanStatusInterval) {
                scanStatusInterval = setInterval(checkScanStatus, 3000);
            }
        } else if (job.status === 'completed') {
            btn.disabled = false;
            btn.textContent = "Scan Context";
            indicator.style.display = 'block';
            indicator.style.background = 'rgba(16, 185, 129, 0.2)';
            indicator.style.color = 'var(--accent)';
            indicator.textContent = `✅ ${job.message}`;
            
            if (scanStatusInterval) {
                clearInterval(scanStatusInterval);
                scanStatusInterval = null;
            }
            
            // Refresh visuals if in settings
            fetch('/get_history').then(r => r.json()).then(data => {
                if (data.visual_memory) {
                    currentVisualMemory = data.visual_memory;
                    if (vmTextarea) vmTextarea.value = currentVisualMemory;
                }
            });
        } else if (job.status === 'error') {
            btn.disabled = false;
            btn.textContent = "Scan Context";
            indicator.style.display = 'block';
            indicator.style.background = 'rgba(239, 68, 68, 0.2)';
            indicator.style.color = 'var(--danger)';
            indicator.textContent = `❌ Error: ${job.message}`;
            
            if (scanStatusInterval) {
                clearInterval(scanStatusInterval);
                scanStatusInterval = null;
            }
        } else {
            btn.disabled = false;
            btn.textContent = "Scan Context";
            indicator.style.display = 'none';
        }
    };

    async function checkScanStatus() {
        try {
            const res = await fetch('/check_scan_status');
            const job = await res.json();
            updateScanUI(job);
        } catch (e) {
            console.warn("Status check failed");
        }
    }

    safeBind('scan-visuals-btn', 'click', async () => {
        const startEl = document.getElementById('set-visual-start');
        const endEl = document.getElementById('set-visual-end');
        const modelEl = document.getElementById('set-model-visual-scan');
        
        const startVal = startEl && startEl.value ? parseInt(startEl.value) : null;
        const endVal = endEl && endEl.value ? parseInt(endEl.value) : null;
        const modelVal = modelEl ? modelEl.value : null;

        const btn = document.getElementById('scan-visuals-btn');
        const ogText = btn.textContent;

        try {
            const res = await fetch('/scan_visuals', { 
                method:'POST', 
                headers:{'Content-Type':'application/json'}, 
                body:JSON.stringify({
                    start: startVal,
                    end: endVal,
                    model: modelVal
                }) 
            });
            const d = await res.json();
            if(d.success) { 
                // Immediate update to show it started
                updateScanUI({status: 'running', message: d.message});
            } else {
                alert("Scan Error: " + d.error);
            }
        } catch(e) { 
            alert("Network Error"); 
            btn.textContent = ogText;
            btn.disabled = false;
        }
    });

    safeBind('save-visuals-btn', 'click', async () => {
        const val = vmTextarea ? vmTextarea.value : "";
        await fetch('/update_visual_memory', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({memory: val}) });
        alert("Memory Saved"); await loadHistory();
    });

    safeBind('sum-enabled', 'click', async (e) => {
        const isChecked = e.target.checked;
        const liveEnabled = document.getElementById('live-sum-enabled')?.checked;
        const opts = document.getElementById('sum-options');
        if (opts) opts.style.display = (isChecked || liveEnabled) ? 'block' : 'none';

        const bgFields1 = document.getElementById('bg-sum-fields');
        const bgFields2 = document.getElementById('bg-sum-prompt-field');
        if (bgFields1) bgFields1.style.display = isChecked ? 'block' : 'none';
        if (bgFields2) bgFields2.style.display = isChecked ? 'block' : 'none';

        if (isChecked) {
            const keepEl = document.getElementById('sum-keep');
            const batchEl = document.getElementById('sum-batch');
            const tempSettings = { 
                recent_turns_to_keep: keepEl ? keepEl.value : 12, 
                batch_size: batchEl ? batchEl.value : 4 
            };
            try {
                const res = await fetch('/check_summary_status', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({settings: tempSettings}) });
                const stats = await res.json();
                if (stats.batches_pending > 0) {
                    if (!confirm(`Summarization will process ${stats.batches_pending} batches. Proceed?`)) {
                        e.target.checked = false; 
                        if (opts) opts.style.display = 'none';
                    }
                }
            } catch(err) { console.error(err); }
        }
    });

    safeBind('rag-enabled', 'click', (e) => {
        const opts = document.getElementById('rag-options');
        if (opts) opts.style.display = e.target.checked ? 'block' : 'none';
    });

    safeBind('tts-enabled', 'click', (e) => {
        const opts = document.getElementById('tts-options');
        if (opts) opts.style.display = e.target.checked ? 'block' : 'none';
    });

    safeBind('set-tts-speed', 'input', (e) => {
        const disp = document.getElementById('tts-speed-val');
        if (disp) disp.textContent = e.target.value + "x";
    });

    safeBind('edit-lorebook-btn', 'click', () => {
        if(modal) modal.style.display = 'none';
        openLargeEditor(null, 'lorebook');
    });

    safeBind('rebuild-index-btn', 'click', async () => {
        const btn = document.getElementById('rebuild-index-btn');
        if (!btn) return;
        const ogText = btn.textContent;
        btn.textContent = "Processing Embeddings..."; btn.disabled = true;
        try {
            const res = await fetch('/rebuild_index', { method: 'POST' });
            const d = await res.json();
            alert(d.message);
        } catch(e) { alert("Rebuild failed."); }
        btn.textContent = ogText; btn.disabled = false;
    });

    safeBind('save-settings-btn', 'click', async () => {
        const arEl = document.getElementById('set-ar');
        const resTierEl = document.querySelector('input[name="res-tier"]:checked');
        const guidedEl = document.getElementById('set-guided');

        const ar = arEl ? arEl.value : "3:4";
        const resTier = resTierEl ? resTierEl.value : "standard";
        const {w,h} = calcPixels(ar, resTier);
        guidedModeEnabled = guidedEl ? guidedEl.checked : false;

        const getVal = (id, def) => { const el = document.getElementById(id); return el ? el.value : def; };
        const getCheck = (id, def) => { const el = document.getElementById(id); return el ? el.checked : def; };

        const fontSize = getVal('set-font-size', '16');
        const bgColor = getVal('set-bg-color', '#121212');

        const payload = {
            venice: { 
                model: getVal('set-model-chat', 'venice-uncensored'), 
                include_venice_system_prompt: getCheck('set-venice-system', true),
                auto_memory_enabled: getCheck('set-auto-memory', false),
                vision_high_res: getCheck('set-vision-high-res', true),
                temperature: parseFloat(getVal('set-temp-chat', 0.8)),
                max_tokens: parseInt(getVal('set-tokens-chat', 4000)),
                reasoning_effort: getVal('set-reasoning-effort', 'medium'),
                frequency_penalty: parseFloat(getVal('set-freq-chat', 0)),
                presence_penalty: parseFloat(getVal('set-pres-chat', 0))
            },
            wfm: {
                model: getVal('ga-wfm', getVal('set-model-wfm', 'venice-uncensored')),
                temperature: parseFloat(getVal('set-temp-wfm', 0.8)),
                context_depth: parseInt(getVal('set-depth-wfm', 10))
            },
            refiner: {
                model: getVal('ga-refiner', getVal('set-model-refiner', 'venice-uncensored')),
                temperature: parseFloat(getVal('set-temp-refiner', 0.3))
            },
            venice_img: { 
                model: getVal('ga-img', getVal('set-model-img', 'qwen3-4b')), 
                visual_scan_model: getVal('ga-visual', getVal('set-model-visual-scan', 'venice-uncensored')),
                context_depth: parseInt(getVal('set-img-depth', 3)) 
            },
            summarizer: { 
                enabled: getCheck('sum-enabled', false), 
                live_summary_enabled: getCheck('live-sum-enabled', false),
                model: getVal('ga-sum', getVal('sum-model', 'qwen3-4b')),
                trigger_threshold_turns: parseInt(getVal('sum-threshold', 12)), 
                batch_size: parseInt(getVal('sum-batch', 4)), 
                recent_turns_to_keep: parseInt(getVal('sum-keep', 12)), 
                system_prompt: getVal('sum-prompt', 'Summarize.'),
                consolidation_model: getVal('ga-cons', getVal('sum-cons-model', 'venice-uncensored')),
                consolidation_prompt: getVal('sum-cons-prompt', 'Summarize.')
            },
            rag: {
                enabled: getCheck('rag-enabled', false),
                k: parseInt(getVal('rag-k', 3)),
                max_chars: parseInt(getVal('rag-max-chars', 1200)),
                min_chars: parseInt(getVal('rag-min-chars', 200)),
                extraction_model: getVal('ga-lore', 'venice-uncensored')
            },
            evaluator_model: getVal('ga-eval', 'venice-uncensored'),
            tts: {
                enabled: getCheck('tts-enabled', false),
                model: getVal('set-tts-model', 'tts-kokoro'),
                voice: getVal('set-tts-model', '').includes('voxtral') ? getVal('mistral-voice-id', '') : getVal('set-tts-voice', 'af_sky'),
                speed: parseFloat(getVal('set-tts-speed', 1.0))
            },
            image_gen: { 
                num_inference_steps: 8,
                width:w, height:h, 
                res_tier: resTier,
                aspect_ratio: ar,
                guided_mode: guidedModeEnabled, 
                art_style: getVal('set-art-style', 'None'), 
                negative_styles: getVal('set-negative-styles', '') 
            },
            interface: { font_size: parseInt(fontSize), bg_color: bgColor }
        };

        await fetch('/save_settings', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });

        // Update local state so UI reflects changes immediately (like TTS buttons and Attachment button)
        window.ttsSettings = payload.tts;
        updateAttachmentButtonVisibility();

        const bannedVal = getVal('set-banned-phrases', '');
        await fetch('/save_banned_phrases', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({banned_phrases: bannedVal}) });

        applyInterfaceSettings(fontSize, bgColor);
        if (modal) modal.style.display = 'none'; 
        await loadHistory();
    });

    safeBind('writeForMeBtn', 'click', async () => {
        if(isGenerating) return;
        const btn = document.getElementById('writeForMeBtn');
        if (!btn) return;

        const currentInput = userInput ? userInput.value.trim() : "";
        if (currentInput) {
            if (!confirm("There is currently text in the prompt field. Do you want to use it as guidance for the Write For Me request?")) {
                return;
            }
        }

        const ogText = btn.textContent;
        btn.textContent = "â³"; btn.disabled = true;
        try {
            const res = await fetch('/write_for_me', { 
                method:'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ guidance: currentInput })
            });
            const d = await res.json();
            if(d.success) { 
                if (userInput) {
                    userInput.value = d.text; 
                    autoExpandTextarea(userInput);
                    userInput.focus();
                }
                lastWfmText = d.text;
                lastWfmUsage = d.debug ? d.debug.usage : null;
            }
            else { alert("Error: " + d.error); }
        } catch(e) { alert("Failed to generate response."); }
        btn.textContent = ogText; btn.disabled = false;
    });

    safeBind('rebuild-all-summaries-btn', 'click', async () => {
        if (!confirm("This will delete ALL current summaries and rebuild them from scratch based on the current threshold and batch size. This will use API tokens. Proceed?")) return;

        const getVal = (id, def) => { const el = document.getElementById(id); return el ? el.value : def; };
        const getCheck = (id, def) => { const el = document.getElementById(id); return el ? el.checked : def; };

        const sumSettings = {
            enabled: getCheck('sum-enabled', false),
            trigger_threshold_turns: parseInt(getVal('sum-threshold', 12)),
            batch_size: parseInt(getVal('sum-batch', 4)),
            recent_turns_to_keep: parseInt(getVal('sum-keep', 12)),
            system_prompt: getVal('sum-prompt', 'Summarize.')
        };
        await fetch('/save_settings', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({summarizer: sumSettings}) });

        const btn = document.getElementById('rebuild-all-summaries-btn');
        if (!btn) return;

        const ogText = btn.textContent;
        btn.textContent = "Rebuilding Summaries... Please wait"; btn.disabled = true;
        try {
            const res = await fetch('/force_summarize', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({mode: 'all'})
            });
            const d = await res.json();
            if(d.success) { alert("Summaries successfully rebuilt!"); await loadHistory(); }
            else { alert("Error: " + d.error); }
        } catch(e) { console.error(e); alert("Failed to rebuild summaries."); }
        btn.textContent = ogText; btn.disabled = false;
    });

    safeBind('open-analytics-btn', 'click', async () => {
        const textMsgCount = chatHistory.filter(m => m.role !== 'system' && !(typeof m.content === 'string' && m.content.startsWith('__IMG_JSON__'))).length;
        const startEl = document.getElementById('ana-start');
        const endEl = document.getElementById('ana-end');
        if (startEl) { startEl.value = 1; startEl.max = textMsgCount; }
        if (endEl) { endEl.value = textMsgCount || 1; endEl.max = textMsgCount; }

        const resDiv = document.getElementById('analytics-results');
        if (resDiv) resDiv.style.display = 'none';

        try {
            const statsRes = await fetch('/get_chat_stats');
            const gData = await statsRes.json();
            const setTxt = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
            setTxt('stat-g-raw', gData.raw_words.toLocaleString());
            setTxt('stat-g-sum1', gData.sum1_words.toLocaleString());
            setTxt('stat-g-sum2', gData.sum2_words.toLocaleString());
            setTxt('stat-p-tot', gData.proj_tot.toLocaleString());
            setTxt('stat-p-raw', gData.proj_raw.toLocaleString());
            setTxt('stat-p-sum', gData.proj_sum.toLocaleString());
            setTxt('stat-p-mem', gData.proj_mem.toLocaleString());
            setTxt('stat-p-sys', gData.proj_sys.toLocaleString());
        } catch (e) {
            console.error("Failed to load global stats", e);
        }

        if (analyticsModal) analyticsModal.style.display = 'block';
    });

    safeBind('open-api-explorer-btn', 'click', () => {
        if (apiExplorerModal) apiExplorerModal.style.display = 'block';
        // Auto-load models tab on open
        loadExplorerTab('models');
    });

    safeBind('close-api-explorer-x', 'click', () => {
        if (apiExplorerModal) apiExplorerModal.style.display = 'none';
    });

    document.querySelectorAll('.explorer-tab').forEach(tab => {
        tab.onclick = () => {
            document.querySelectorAll('.explorer-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            loadExplorerTab(tab.dataset.tab);
        };
    });

    async function loadExplorerTab(endpoint) {
        const results = document.getElementById('explorer-results');
        if (!results) return;
        results.innerHTML = `<div style="text-align:center; padding:50px;"><div class="spinner" style="width:30px; height:30px; margin:0 auto;"></div><br>Fetching ${endpoint} from Venice...</div>`;

        try {
            const res = await fetch(`/venice/discovery/${endpoint}`);
            const data = await res.json();

            if (data.error) {
                results.innerHTML = `<div style="color:var(--danger); padding:20px;">Error: ${data.error}</div>`;
                return;
            }

            if (endpoint === 'models') {
                renderDiscoveredModels(data.data, results);
            } else {
                results.innerHTML = `<pre style="font-family:monospace; font-size:0.85em; white-space:pre-wrap; color:#ccc;">${JSON.stringify(data, null, 2)}</pre>`;
            }
        } catch (e) {
            results.innerHTML = `<div style="color:var(--danger); padding:20px;">Failed to connect to backend.</div>`;
        }
    }

    function renderDiscoveredModels(models, container) {
        container.innerHTML = `
            <div style="margin-bottom:15px; display:flex; justify-content:space-between; align-items:center;">
                <h3 style="margin:0; color:var(--accent);">Available Models (${models.length})</h3>
                <input type="text" id="explorer-search" placeholder="Search models..." style="background:#222; border:1px solid #444; color:white; padding:6px 12px; border-radius:6px; font-size:0.9em;">
            </div>
            <div id="explorer-list" style="display:flex; flex-direction:column; gap:10px;"></div>
        `;

        const list = document.getElementById('explorer-list');
        const searchInput = document.getElementById('explorer-search');

        const drawList = (filter = "") => {
            list.innerHTML = "";
            const filtered = models.filter(m => 
                m.id.toLowerCase().includes(filter.toLowerCase()) || 
                (m.name && m.name.toLowerCase().includes(filter.toLowerCase()))
            );

            filtered.forEach(m => {
                const card = document.createElement('div');
                card.style.cssText = "background:#1a1a1a; border:1px solid #333; border-radius:8px; padding:12px; display:flex; justify-content:space-between; align-items:center; transition:border-color 0.2s;";
                card.onmouseenter = () => card.style.borderColor = "var(--primary)";
                card.onmouseleave = () => card.style.borderColor = "#333";

                const info = document.createElement('div');
                info.style.flex = "1";
                
                const spec = m.model_spec || {};
                const modelName = spec.name || m.name || m.id;

                let pricingStr = "Free/Unknown";
                if (spec.pricing) {
                    pricingStr = `${spec.pricing.input?.usd || 0} / ${spec.pricing.output?.usd || 0}`;
                } else if (m.pricePer1MInputTokens) {
                    pricingStr = `${m.pricePer1MInputTokens} / ${m.pricePer1MOutputTokens}`;
                }

                const contextVal = spec.availableContextTokens || m.contextWindowSize;
                const contextStr = contextVal ? contextVal.toLocaleString() : "N/A";

                info.innerHTML = `
                    <div style="font-weight:bold; color:white;">${modelName}</div>
                    <div style="font-size:0.75em; color:var(--text-dim); font-family:monospace;">${m.id}</div>
                    <div style="font-size:0.8em; color:var(--accent); margin-top:4px;">Context: ${contextStr} • Price: ${pricingStr}</div>
                `;

                const btn = document.createElement('button');
                btn.className = "msg-btn";
                btn.style.cssText = "background:var(--primary); color:white; border:none; padding:6px 12px;";
                btn.textContent = "Import";
                btn.onclick = () => importDiscoveredModel(m);

                card.appendChild(info);
                card.appendChild(btn);
                list.appendChild(card);
            });
        };

        searchInput.oninput = (e) => drawList(e.target.value);
        drawList();
    }

    async function importDiscoveredModel(m) {
        const category = prompt("Which category should this model be added to?", "Imported Models");
        if (category === null) return;

        const spec = m.model_spec || {};
        const capabilities = spec.capabilities || m.capabilities || {};

        let pricingStr = "N/A";
        if (spec.pricing) {
            pricingStr = `${spec.pricing.input?.usd || 0} / ${spec.pricing.output?.usd || 0}`;
        } else if (m.pricePer1MInputTokens) {
            pricingStr = `${m.pricePer1MInputTokens} / ${m.pricePer1MOutputTokens}`;
        }

        const payload = {
            id: m.id,
            name: spec.name || m.name || m.id,
            traits: (spec.traits ? spec.traits.join(", ") : null) || m.type || "Universal",
            pricing: pricingStr,
            context: (spec.availableContextTokens || m.contextWindowSize || "N/A").toLocaleString(),
            vision: capabilities.supportsVision || capabilities.supportsMultipleImages || false,
            description: spec.description || "Imported via API Explorer.",
            tags: m.tags || []
        };

        try {
            const res = await fetch(`/venice/import_model?category=${encodeURIComponent(category)}`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
            const d = await res.json();
            if (d.success) {
                alert(`Successfully added ${payload.name} to ${category}!`);
                // Refresh local models cache
                const mRes = await fetch('/venice_models');
                availableModels = await mRes.json();
            } else {
                alert(d.message || "Failed to import model.");
            }
        } catch (e) {
            alert("Error importing model.");
        }
    }

    safeBind('run-analytics-btn', 'click', async () => {
        await runAnalytics();
    });

    safeBind('close-analytics-x', 'click', () => {
        if (analyticsModal) analyticsModal.style.display = 'none';
    });

    // --- LORE EXTRACTOR LOGIC ---
    safeBind('open-lore-extractor-btn', 'click', () => {
        const textMsgCount = chatHistory.filter(m => m.role !== 'system' && !(typeof m.content === 'string' && m.content.startsWith('__IMG_JSON__'))).length;
        const startEl = document.getElementById('le-start');
        const endEl = document.getElementById('le-end');
        if (startEl) { startEl.value = 1; startEl.max = textMsgCount; }
        if (endEl) { endEl.value = textMsgCount || 1; endEl.max = textMsgCount; }

        const outputTa = document.getElementById('le-output');
        if (outputTa) outputTa.value = '';

        const saveBtn = document.getElementById('le-save-btn');
        if (saveBtn) saveBtn.style.display = 'none';

        if (loreExtractorModal) loreExtractorModal.style.display = 'block';
    });

    safeBind('close-lore-extractor-x', 'click', () => {
        if (loreExtractorModal) loreExtractorModal.style.display = 'none';
    });

    safeBind('open-cache-debugger-btn', 'click', async () => {
        const modal = document.getElementById('cache-debug-modal');
        if (modal) {
            modal.style.display = 'block';
            document.body.style.overflow = 'hidden';
        }
        await loadCacheDebug();
    });

    safeBind('close-cache-debug-x', 'click', () => {
        const modal = document.getElementById('cache-debug-modal');
        if (modal) {
            modal.style.display = 'none';
            document.body.style.overflow = '';
        }
    });

    safeBind('open-last-io-btn', 'click', async () => {
        const modal = document.getElementById('last-io-modal');
        if (modal) {
            modal.style.display = 'block';
            document.body.style.overflow = 'hidden'; // Prevent background scrolling
        }
        await loadLastIO();
    });

    safeBind('close-last-io-x', 'click', () => {
        const modal = document.getElementById('last-io-modal');
        if (modal) {
            modal.style.display = 'none';
            document.body.style.overflow = ''; // Restore background scrolling
        }
    });

    // --- LEDGER LOGIC ---
    safeBind('open-ledger-btn', 'click', async () => {
        const modal = document.getElementById('ledger-modal');
        if (modal) {
            modal.style.display = 'block';
            document.body.style.overflow = 'hidden';
        }
        await loadLedger();
    });

    safeBind('close-ledger-x', 'click', () => {
        const modal = document.getElementById('ledger-modal');
        if (modal) {
            modal.style.display = 'none';
            document.body.style.overflow = '';
        }
    });

    safeBind('clear-ledger-btn', 'click', async () => {
        if (!confirm("Are you sure you want to permanently delete all API call history from the ledger?")) return;
        try {
            await fetch('/clear_ledger', { method: 'POST' });
            await loadLedger();
        } catch (e) {
            alert("Error clearing ledger.");
        }
    });

    async function loadLedger() {
        const tBody = document.getElementById('ledger-tbody');
        const summaryDiv = document.getElementById('ledger-summary');
        if (!tBody || !summaryDiv) return;

        tBody.innerHTML = '<tr><td colspan="6" style="text-align:center;">Loading...</td></tr>';
        
        try {
            const res = await fetch('/get_ledger');
            const data = await res.json();
            
            // Build Summary
            const s = data.summary;
            let summaryHTML = `
                <div style="display:flex; justify-content:space-between; margin-bottom:15px; flex-wrap:wrap; gap:10px;">
                    <div style="background:var(--bg-lighter); padding:10px; border-radius:4px; flex:1; min-width:150px;">
                        <strong>Total Calls:</strong><br>${s.total_calls}
                    </div>
                    <div style="background:var(--bg-lighter); padding:10px; border-radius:4px; flex:1; min-width:150px;">
                        <strong>Total Est. Cost:</strong><br>$${s.total_estimated.toFixed(4)}
                    </div>
                    <div style="background:var(--bg-lighter); padding:10px; border-radius:4px; flex:1; min-width:150px;">
                        <strong>Total Actual Cost:</strong><br>$${s.total_actual.toFixed(4)}
                    </div>
                    <div style="background:var(--bg-lighter); padding:10px; border-radius:4px; flex:1; min-width:150px;">
                        <strong>Avg Cache Hit:</strong><br>${s.avg_cache_hit_rate}%
                    </div>
                </div>
                <div style="font-size:0.9em;">
                    <strong>Cost by Feature:</strong><br>
            `;
            for (const [feat, stats] of Object.entries(s.feature_breakdown)) {
                summaryHTML += `<span style="display:inline-block; margin-right:15px; background:rgba(255,255,255,0.05); padding:2px 6px; border-radius:3px;">${feat}: $${stats.actual.toFixed(4)}</span>`;
            }
            summaryHTML += `</div>`;
            summaryDiv.innerHTML = summaryHTML;

            // Build Table (reversed so newest is on top)
            const calls = data.calls.slice().reverse();
            tBody.innerHTML = '';
            
            if (calls.length === 0) {
                tBody.innerHTML = '<tr><td colspan="6" style="text-align:center;">No API calls logged yet.</td></tr>';
                return;
            }

            for (const c of calls) {
                const tr = document.createElement('tr');
                
                const timeStr = new Date(c.timestamp).toLocaleString();
                
                // Color code actual vs estimated
                let costColor = '';
                const diff = (c.actual_cost || 0) - (c.estimated_cost || 0);
                if (diff > 0.001) costColor = 'color:var(--red);'; // Significantly higher than estimated
                else if (diff < -0.001) costColor = 'color:var(--accent);'; // Lower than estimated (cache hit)
                
                let actualCostStr = c.actual_cost !== null ? `$${c.actual_cost.toFixed(4)}` : 'N/A';
                if (c.actual_cost !== null) {
                    actualCostStr = `<span style="${costColor}">${actualCostStr}</span>`;
                }

                let estCostStr = c.estimated_cost !== null ? `$${c.estimated_cost.toFixed(4)}` : 'N/A';

                tr.innerHTML = `
                    <td style="font-size:0.8em; color:var(--text-dim);">${timeStr}</td>
                    <td><strong>${c.feature}</strong><br><span style="font-size:0.8em; color:var(--text-dim);">${c.model}</span></td>
                    <td style="text-align:right;">${c.prompt_tokens.toLocaleString()}<br><span style="font-size:0.8em; color:var(--text-dim);">${c.completion_tokens.toLocaleString()} out</span></td>
                    <td style="text-align:right;">
                        ${c.cached_tokens.toLocaleString()}<br>
                        <span style="font-size:0.8em; color:${c.cache_hit_rate > 50 ? 'var(--accent)' : 'var(--text-dim)'};">${c.cache_hit_rate}%</span>
                    </td>
                    <td style="text-align:right; font-family:monospace;">${estCostStr}</td>
                    <td style="text-align:right; font-family:monospace; font-weight:bold;">${actualCostStr}</td>
                `;
                tBody.appendChild(tr);
            }

        } catch (e) {
            tBody.innerHTML = `<tr><td colspan="6" style="text-align:center; color:red;">Error fetching ledger: ${e.message}</td></tr>`;
        }
    }

    async function handleDeleteMessage(index) {
        if (!confirm("Remove this message from the chat? This will remove it from the conversation context.")) return;

        // Structure check: If we delete a message, we might end up with two consecutive roles.
        // We'll remove it and then check the remaining list.
        chatHistory.splice(index, 1);

        // If a deletion results in two consecutive roles (e.g. User, [Deleted Assistant], User)
        // most APIs will actually handle this fine, but for clean context, we can optionally merge them.
        // For now, we just remove and save.

        // Also clear from diff mode set if present
        diffModeMsgs.delete(index);

        await saveHistory();
        renderChat();
    }

    async function loadLastIO() {
        const reqPre = document.getElementById('last-io-request');
        const resPre = document.getElementById('last-io-response');
        const epSpan = document.getElementById('last-io-endpoint');
        const timeSpan = document.getElementById('last-io-time');
        if (!reqPre || !resPre) return;

        reqPre.textContent = "Loading...";
        resPre.textContent = "Loading...";
        if(epSpan) epSpan.textContent = "Loading...";
        if(timeSpan) timeSpan.textContent = "Loading...";

        try {
            const res = await fetch('/get_last_io');
            const d = await res.json();
            if(epSpan) epSpan.textContent = d.endpoint || "None";
            if(timeSpan) {
                if (d.timestamp) {
                    const t = new Date(d.timestamp);
                    timeSpan.textContent = t.toLocaleTimeString() + " " + t.toLocaleDateString();
                } else {
                    timeSpan.textContent = "None";
                }
            }
            reqPre.textContent = JSON.stringify(d.request, null, 2) || "No request logged yet.";
            resPre.textContent = JSON.stringify(d.response, null, 2) || "No response logged yet.";
        } catch(e) {
            reqPre.textContent = "Error fetching IO data.";
        }
    }

    safeBind('view-tts-log-btn', 'click', async () => {
        const modal = document.getElementById('tts-log-modal');
        const display = document.getElementById('tts-log-display');
        if (modal) modal.style.display = 'block';
        if (display) display.textContent = "Fetching log...";

        try {
            const res = await fetch('/get_last_tts_log');
            const data = await res.json();
            display.textContent = JSON.stringify(data, null, 2);
        } catch (e) {
            display.textContent = "Error fetching TTS log: " + e.message;
        }
    });

    safeBind('refresh-cache-debug-btn', 'click', async () => {
        await loadCacheDebug();
    });

    async function loadCacheDebug() {
        const resultsDiv = document.getElementById('cache-debug-results');
        if (!resultsDiv) return;
        resultsDiv.innerHTML = 'Analyzing payloads...';

        try {
            const res = await fetch('/debug_cache');
            const d = await res.json();
            if (d.error) {
                resultsDiv.innerHTML = `<span style="color:var(--danger)">${d.error}</span>`;
                return;
            }

            let html = `<div style="margin-bottom:15px; border-bottom:1px solid #333; padding-bottom:10px;">`;
            html += `<strong>Previous Payload Messages:</strong> <span style="color:var(--text-main)">${d.prev_len}</span>\n`;
            html += `<strong>Last Payload Messages:</strong> <span style="color:var(--text-main)">${d.last_len}</span>\n</div>`;

            if (!d.divergence) {
                html += `<div style="color:var(--accent); font-weight:bold; font-size:1.1em; margin-bottom:5px;">✅ PERFECT MATCH!</div>`;
                html += `<span style="color:var(--text-dim)">The previous payload is an exact prefix of the last payload. Token caching should have hit successfully for the entire previous context window.</span>`;
            } else {
                const div = d.divergence;
                html += `<div style="color:var(--danger); font-weight:bold; font-size:1.1em; margin-bottom:5px;">❌ CACHE BUSTED AT MESSAGE INDEX [${div.message_index}]</div>`;

                const cachedTokensNote = `<div style="background:rgba(16, 185, 129, 0.1); border-left:3px solid var(--accent); padding:8px; margin:10px 0; color:#ddd;"><strong>Optimal Caching:</strong> All messages before index ${div.message_index} were successfully cached!</div>`;

                if (div.type === 'parameter_change') {
                    html = html.replace(`❌ CACHE BUSTED AT MESSAGE INDEX [${div.message_index}]`, `❌ CACHE BUSTED BY PARAMETER CHANGE`);
                    html += `<span style="color:var(--text-dim)">Parameter changed: <strong style="color:white">${div.parameter}</strong>\n`;
                    html += `Expected (Previous): <span style="color:var(--primary)">${JSON.stringify(div.expected)}</span>\n`;
                    html += `Got (Last): <span style="color:var(--danger)">${JSON.stringify(div.got)}</span>\n\n`;
                    html += `(Any change to generation parameters, models, or system settings alters the hash and invalidates the cache entirely!)</span>`;
                } else if (div.type === 'addition') {
                    html += cachedTokensNote;
                    html += `<span style="color:var(--text-dim)">Reason: ${div.reason}\n(Note: This usually means the context grew perfectly, so caching should still apply to everything before this index.)</span>`;
                } else if (div.type === 'deletion') {
                    html += cachedTokensNote;
                    html += `<span style="color:var(--text-dim)">Reason: ${div.reason}\n(Messages were removed or truncated, likely due to a summarization event or manual deletion.)</span>`;
                } else if (div.type === 'role_change') {
                    html += cachedTokensNote;
                    html += `<span style="color:var(--text-dim)">Role changed from '<span style="color:white">${div.expected}</span>' to '<span style="color:white">${div.got}</span>'.\n(If this was at the very end of the previous payload, it might be an ephemeral RAG prompt being replaced by an assistant response, which is normal.)</span>`;
                } else if (div.type === 'content_change') {
                    html += cachedTokensNote;
                    html += `Message Role: <strong style="color:var(--primary)">${div.role}</strong>\n`;
                    html += `Mismatch detected at character <strong style="color:var(--danger)">${div.diff_idx}</strong> of this message's JSON string.\n\n`;

                    html += `<div style="background:#222; padding:10px; border-radius:4px; margin-bottom:10px;">`;
                    html += `<strong style="color:var(--text-dim)">PREVIOUS PAYLOAD (What the cache expected):</strong>\n`;

                    const escapeHtml = (t) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

                    const pBefore = escapeHtml(div.snippet_prev.substring(0, 50));
                    const pAfter = escapeHtml(div.snippet_prev.substring(50));
                    html += `...${pBefore}<span style="background:rgba(239, 68, 68, 0.4); color:#fff; border-bottom:2px solid #ef4444;">${pAfter.substring(0,1) || ' '}</span>${pAfter.substring(1)}...\n`;
                    html += `</div>`;

                    html += `<div style="background:#222; padding:10px; border-radius:4px;">`;
                    html += `<strong style="color:var(--text-dim)">LAST PAYLOAD (What was actually sent):</strong>\n`;
                    const lBefore = escapeHtml(div.snippet_last.substring(0, 50));
                    const lAfter = escapeHtml(div.snippet_last.substring(50));
                    html += `...${lBefore}<span style="background:rgba(16, 185, 129, 0.4); color:#fff; border-bottom:2px solid #10b981;">${lAfter.substring(0,1) || ' '}</span>${lAfter.substring(1)}...\n`;
                    html += `</div>`;

                    html += `\n<span style="color:#888; font-size:0.9em;">Tip: Look at the highlighted character above to see exactly what changed. Minor edits, summary rebuilds, or visual memory scans can cause this.</span>`;
                }
            }
            resultsDiv.innerHTML = html;
        } catch (e) {
            resultsDiv.innerHTML = `<span style="color:var(--danger)">Failed to fetch cache debug data. The server might be unreachable or logs are corrupted.</span>`;
        }
    }

    safeBind('run-lore-extractor-btn', 'click', async () => {
        const startNum = parseInt(document.getElementById('le-start')?.value);
        const endNum = parseInt(document.getElementById('le-end')?.value);
        const batchSize = parseInt(document.getElementById('le-batch')?.value) || 20;

        if (isNaN(startNum) || isNaN(endNum) || startNum > endNum) {
            alert("Invalid message range."); return;
        }

        const btn = document.getElementById('run-lore-extractor-btn');
        const outputTa = document.getElementById('le-output');
        const saveBtn = document.getElementById('le-save-btn');

        const ogText = btn.textContent;
        btn.textContent = "Extracting..."; btn.disabled = true;
        if(outputTa) outputTa.value = "Scanning messages and writing lore. This may take a minute depending on batch size...";

        try {
            const res = await fetch('/extract_lore', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ start: startNum, end: endNum, batch_size: batchSize })
            });
            const d = await res.json();

            if (d.success) {
                if(outputTa) {
                    outputTa.value = d.lore || "No significant new lore found in this range.";
                }
                if (saveBtn && d.lore) {
                    saveBtn.style.display = 'block';
                } else if (saveBtn) {
                    saveBtn.style.display = 'none';
                }
            } else {
                alert("Extraction Error: " + (d.error || "Unknown error"));
                if(outputTa) outputTa.value = "";
            }
        } catch(e) {
            console.error("Extraction failed:", e);
            alert("Failed to extract lore. Check console.");
            if(outputTa) outputTa.value = "";
        }
        btn.textContent = ogText; btn.disabled = false;
    });

    safeBind('le-save-btn', 'click', async () => {
        const outputTa = document.getElementById('le-output');
        if (!outputTa || !outputTa.value.trim()) return;

        const newLore = outputTa.value.trim();
        const btn = document.getElementById('le-save-btn');
        const ogText = btn.textContent;
        btn.textContent = "Saving..."; btn.disabled = true;

        try {
            // Get current lorebook
            const lr = await fetch('/get_lorebook');
            const lData = await lr.json();
            const currentLore = lData.text || "";

            // Append
            const combinedLore = currentLore ? currentLore + "\n\n" + newLore : newLore;

            // Save
            await fetch('/save_lorebook', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text:combinedLore}) });

            // Rebuild Index
            btn.textContent = "Rebuilding Index...";
            const rr = await fetch('/rebuild_index', { method: 'POST' });
            const rData = await rr.json();

            alert("Lore successfully appended and vector index rebuilt!");
            if (loreExtractorModal) loreExtractorModal.style.display = 'none';
        } catch(e) {
            alert("Failed to save and rebuild.");
            console.error(e);
        }
        btn.textContent = ogText; btn.disabled = false;
    });

// --- STORY PIPELINE LOGIC ---
document.querySelectorAll('.pipe-select').forEach(sel => {
    sel.addEventListener('change', (e) => {
        const customInput = document.getElementById(e.target.id + '-custom');
        if (customInput) {
            customInput.style.display = e.target.value === 'custom' ? 'block' : 'none';
        }
    });
});

// --- STORY PIPELINE LOGIC (INTEGRATED) ---
const newPipelineBtn = document.getElementById('new-pipeline-btn');
const pipelineSetupModal = document.getElementById('pipeline-setup-modal');
const closePipelineSetupX = document.getElementById('close-pipeline-setup-x');
const startPipelineBtn = document.getElementById('start-pipeline-btn');
const pipelineLockBtn = document.getElementById('pipeline-lock-btn');
const pipelineSaveDocBtn = document.getElementById('pipeline-save-doc-btn');
const openPipelinePanelBtn = document.getElementById('open-pipeline-panel-btn');
let currentPipelinePhase = 'architect';

if (newPipelineBtn) {
    newPipelineBtn.addEventListener('click', () => {
        const sidebar = document.getElementById('sidebar');
        if (sidebar) sidebar.classList.add('hidden');
        if (pipelineSetupModal) {
            pipelineSetupModal.style.display = 'flex';
            // Trigger model population
            if (typeof filterAllModelSelectors === 'function') filterAllModelSelectors();
            
            // Re-bind custom toggles just in case
            document.querySelectorAll('.pipe-select').forEach(sel => {
                sel.addEventListener('change', (e) => {
                    const customInput = document.getElementById(sel.id + '-custom');
                    if (customInput) {
                        customInput.style.display = e.target.value === 'custom' ? 'block' : 'none';
                        if (e.target.value === 'custom') customInput.focus();
                    }
                });
            });
        }
    });
}
if (closePipelineSetupX) closePipelineSetupX.addEventListener('click', () => pipelineSetupModal.style.display = 'none');

if (startPipelineBtn) {
    startPipelineBtn.addEventListener('click', async () => {
        const concept = document.getElementById('pipe-concept').value.trim();
        if (!concept) {
            alert("Core Concept is required.");
            return;
        }
        
        const getSetting = (id) => {
            const el = document.getElementById(id);
            if (!el) return '';
            const val = el.value;
            if (val === 'custom') {
                const cEl = document.getElementById(id + '-custom');
                return cEl ? cEl.value.trim() : '';
            }
            return val;
        };

        const settings = {
            length: getSetting('pipe-length'),
            detail: getSetting('pipe-detail'),
            genre: getSetting('pipe-genre'),
            tone: getSetting('pipe-tone'),
            themes: getSetting('pipe-themes'),
            setting: getSetting('pipe-setting'),
            pov: getSetting('pipe-pov'),
            pacing: getSetting('pipe-pacing'),
            nsfw: getSetting('pipe-nsfw')
        };
        
        if (pipelineSetupModal) pipelineSetupModal.style.display = 'none';
        
        try {
            const res = await fetch('/create_pipeline_chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ concept, settings })
            });
            const data = await res.json();
            if (data.success) {
                await loadSidebar();
                const sbRes = await fetch('/sidebar_data');
                const sbData = await sbRes.json();
                if (sbData.active_chat) {
                    await loadHistory();
                    // The backend already seeded the new pipeline with the concept/settings
                    // as the first user message. Passing an empty message tells /chat to
                    // generate from that existing last user message instead of appending
                    // a second automatic user message.
                    triggerGen("");
                }
            } else {
                alert("Error: " + data.error);
            }
        } catch (e) { console.error(e); }
    });
}

// Full-screen Modal Toggle
if (openPipelinePanelBtn) {
    openPipelinePanelBtn.addEventListener('click', () => {
        const docModal = document.getElementById('pipeline-doc-modal');
        if (docModal) {
            docModal.style.display = 'flex';
            document.body.style.overflow = 'hidden';
            
            // Sync editor content
            const editor = document.getElementById('blueprint-editor');
            if (editor) {
                // If it's a pipeline chat, history might have the blueprint
                // We'll trust the current value in editor or fetch it
            }
        }
    });
}

safeBind('close-pipeline-doc-x', 'click', () => {
    const docModal = document.getElementById('pipeline-doc-modal');
    if (docModal) {
        docModal.style.display = 'none';
        document.body.style.overflow = '';
    }
});

// Lock Phase
safeBind('lock-scribe-btn', 'click', async () => {
    try {
        const res = await fetch('/toggle_pipeline_phase', {method: 'POST'});
        const data = await res.json();
        if (data.success) {
            currentPipelinePhase = data.new_phase;
            const badge = document.getElementById('pipeline-status-badge');
            const modeLabel = document.getElementById('pipeline-mode-label');
            if (badge) {
                badge.textContent = currentPipelinePhase.toUpperCase();
                badge.style.background = currentPipelinePhase === 'scribe' ? 'var(--purple)' : 'var(--accent)';
            }
            if (modeLabel) modeLabel.textContent = currentPipelinePhase.charAt(0).toUpperCase() + currentPipelinePhase.slice(1);
            
            const lockBtn = document.getElementById('lock-scribe-btn');
            const editor = document.getElementById('blueprint-editor');
            if (currentPipelinePhase === 'scribe') {
                if (lockBtn) lockBtn.textContent = '← Back to Architect';
                if (editor) editor.readOnly = true;
            } else {
                if (lockBtn) lockBtn.textContent = 'Lock & Scribe';
                if (editor) editor.readOnly = false;
            }

            // Reload the active pipeline track immediately so the visible chat switches
            // between architect_messages and scribe_messages without needing a page refresh.
            await loadHistory();
        }
    } catch (e) { console.error(e); }
});

// --- LINE-BASED DIFF LOGIC (Simplified Myers-ish) ---
let previousBlueprint = '';

function updateBlueprintDiff(oldText, newText) {
    const diffView = document.getElementById('diff-view-container');
    if (!diffView) return;

    const oldLines = oldText.split('\n');
    const newLines = newText.split('\n');
    
    // Basic greedy line diff
    let html = '';
    let i = 0, j = 0;

    while (i < oldLines.length || j < newLines.length) {
        if (i < oldLines.length && j < newLines.length && oldLines[i] === newLines[j]) {
            html += `<div class="diff-line-equal">${escapeHtml(oldLines[i]) || '&nbsp;'}</div>`;
            i++; j++;
        } else {
            // Check for potential resync (look ahead)
            let foundResync = false;
            let lookAhead = 10; // lines

            for (let k = 1; k <= lookAhead; k++) {
                // Deletion
                if (i + k < oldLines.length && oldLines[i + k] === newLines[j]) {
                    for (let m = 0; m < k; m++) {
                        html += `<div class="diff-line-remove">- ${escapeHtml(oldLines[i + m]) || '&nbsp;'}</div>`;
                    }
                    i += k;
                    foundResync = true;
                    break;
                }
                // Addition
                if (j + k < newLines.length && oldLines[i] === newLines[j + k]) {
                    for (let m = 0; m < k; m++) {
                        html += `<div class="diff-line-add">+ ${escapeHtml(newLines[j + m]) || '&nbsp;'}</div>`;
                    }
                    j += k;
                    foundResync = true;
                    break;
                }
            }

            if (!foundResync) {
                // If no resync, it's a replacement or end of text
                if (i < oldLines.length && j < newLines.length) {
                    html += `<div class="diff-line-remove">- ${escapeHtml(oldLines[i]) || '&nbsp;'}</div>`;
                    html += `<div class="diff-line-add">+ ${escapeHtml(newLines[j]) || '&nbsp;'}</div>`;
                    i++; j++;
                } else if (i < oldLines.length) {
                    html += `<div class="diff-line-remove">- ${escapeHtml(oldLines[i]) || '&nbsp;'}</div>`;
                    i++;
                } else if (j < newLines.length) {
                    html += `<div class="diff-line-add">+ ${escapeHtml(newLines[j]) || '&nbsp;'}</div>`;
                    j++;
                }
            }
        }
    }
    
    diffView.innerHTML = html || '<div style="color:#666; font-style:italic; padding: 20px;">No changes yet...</div>';
}

function escapeHtml(text) {
    if (!text) return '';
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

// Bind Toggles
safeBind('view-edit-btn', 'click', () => {
    document.getElementById('view-edit-btn').classList.add('active');
    document.getElementById('view-diff-btn').classList.remove('active');
    document.getElementById('edit-view-container').style.display = 'block';
    document.getElementById('diff-view-container').style.display = 'none';
});

safeBind('view-diff-btn', 'click', () => {
    document.getElementById('view-diff-btn').classList.add('active');
    document.getElementById('view-edit-btn').classList.remove('active');
    document.getElementById('edit-view-container').style.display = 'none';
    document.getElementById('diff-view-container').style.display = 'block';
    
    // Compute diff on demand: compare saved blueprint vs current editor content
    const editor = document.getElementById('blueprint-editor');
    const currentText = editor ? editor.value : '';
    updateBlueprintDiff(previousBlueprint, currentText);
});

// Save Doc
safeBind('save-blueprint-btn', 'click', async () => {
    const editor = document.getElementById('blueprint-editor');
    const saveBtn = document.getElementById('save-blueprint-btn');
    if (!editor || !saveBtn) return;
    try {
        const res = await fetch('/update_blueprint', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ blueprint: editor.value })
        });
        if (res.ok) {
            const ogText = saveBtn.textContent;
            saveBtn.textContent = 'Saved!';
            saveBtn.style.background = 'var(--accent)';
            setTimeout(() => {
                saveBtn.textContent = ogText;
                saveBtn.style.background = 'var(--primary)';
            }, 2000);
        }
    } catch(e) {}
});
});