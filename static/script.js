document.addEventListener('DOMContentLoaded', () => {
    let chatHistory = [];
    let summaries = [];
    let currentVisualMemory = "";
    let isGenerating = false;
    let guidedModeEnabled = false;
    let activeEditIndex = -1;
    let activeEditType = 'message';
    let runningInput = 0;
    let runningOutput = 0;
    let diffModeMsgs = new Set();

    const chatbox = document.getElementById('chatbox');
    const userInput = document.getElementById('userInput');
    const modal = document.getElementById('settingsModal');
    const editOverlay = document.getElementById('edit-overlay');
    const editTextarea = document.getElementById('edit-textarea');
    const guidedOverlay = document.getElementById('guided-overlay');
    const guidedInput = document.getElementById('guided-input');
    const vmTextarea = document.getElementById('visual-memory-text');
    const analyticsModal = document.getElementById('analytics-modal');
    const loreExtractorModal = document.getElementById('lore-extractor-modal');

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

    const init = async () => {
        try {
            const mRes = await fetch('/venice_models');
            availableModels = await mRes.json();

            // Populate Auto-Rename model selector
            const renameSelect = document.getElementById('rename-model-select');
            if (renameSelect && availableModels) {
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
                    renameSelect.appendChild(optgroup);
                }
            }

            const settingsRes = await fetch('/get_settings');
            const d = await settingsRes.json();
            if(d && d.interface) {
                applyInterfaceSettings(d.interface.font_size, d.interface.bg_color);
            }
            if(d && d.tts) {
                window.ttsSettings = d.tts;
            }
            await loadHistory();
            await loadSidebar();
            updateAttachmentButtonVisibility();
        } catch (e) {
            console.error("Initialization failed:", e);
        }
    };

    init();

    // --- TTS LOGIC ---
    let currentTTS = {
        msgIndex: -1,
        audio: null,
        stopFlag: false,
        unlocked: false // Track if we've primed the iOS audio engine
    };

    // Helper to prime the audio engine (Crucial for iOS/Safari)
    function unlockIOSAudio() {
        if (currentTTS.unlocked) return;
        const silentSrc = 'data:audio/wav;base64,UklGRigAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=';
        const audio = new Audio(silentSrc);
        audio.play().then(() => {
            currentTTS.unlocked = true;
            console.log("iOS Audio Engine Unlocked");
        }).catch(e => console.warn("Audio unlock failed", e));
    }

    async function handleTTS(index) {
        // SYNCHRONOUS: This happens the instant you tap
        unlockIOSAudio(); 

        const msg = chatHistory[index];
        if (!msg || !msg.content) return;

        const btn = document.querySelector(`#msg-${index} .tts-btn`);

        // Handle Stop Button Logic
        if (currentTTS.msgIndex === index) {
            currentTTS.stopFlag = true;
            if (currentTTS.audio) {
                currentTTS.audio.pause();
                currentTTS.audio = null;
            }
            currentTTS.msgIndex = -1;
            if (btn) btn.textContent = '▶ Narrate';
            return;
        }

        // Stop any other playing messages
        if (currentTTS.msgIndex !== -1) {
            currentTTS.stopFlag = true;
            if (currentTTS.audio) currentTTS.audio.pause();
            const oldBtn = document.querySelector(`#msg-${currentTTS.msgIndex} .tts-btn`);
            if (oldBtn) oldBtn.textContent = '▶ Narrate';
        }

        currentTTS.msgIndex = index;
        currentTTS.stopFlag = false;
        if (btn) btn.textContent = '⌛...';

        // Clean and chunk text locally
        const textContent = Array.isArray(msg.content) ? (msg.content.find(p => p.type === 'text')?.text || "") : msg.content;
        const cleanText = (textContent || "").replace(/<think>[\s\S]*?<\/think>/gi, '').replace(/<[^>]+>/g, '').replace(/[*_`#]/g, '').replace(/\[.*?\]\(.*?\)/g, '').trim();

        if (!cleanText) {
            if (btn) btn.textContent = '▶ Narrate';
            currentTTS.msgIndex = -1;
            return;
        }

        const chunks = [];
        const maxChars = 3000;
        let remaining = cleanText;
        while(remaining.length > 0) {
            if (remaining.length <= maxChars) {
                chunks.push(remaining);
                remaining = "";
            } else {
                let splitIdx = remaining.lastIndexOf('\n\n', maxChars);
                if (splitIdx === -1) splitIdx = remaining.lastIndexOf('\n', maxChars);
                if (splitIdx === -1) splitIdx = remaining.lastIndexOf('.', maxChars);
                if (splitIdx === -1) splitIdx = maxChars;
                chunks.push(remaining.substring(0, splitIdx + 1).trim());
                remaining = remaining.substring(splitIdx + 1).trim();
            }
        }

        if (btn) btn.textContent = '⏹ Stop';

        const fetchAudio = async (textChunk) => {
            try {
                const res = await fetch('/tts', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: textChunk })
                });
                if (res.ok) {
                    const balance = res.headers.get('x-venice-balance-usd');
                    if(balance) updateBalanceDisplay(balance);
                    return await res.blob();
                }
            } catch(e) { console.error("Fetch failed", e); }
            return null;
        };

        let nextBlob = null;

        for (let i = 0; i < chunks.length; i++) {
            if (currentTTS.stopFlag) break;

            let currentBlob = (i === 0) ? await fetchAudio(chunks[i]) : nextBlob;
            if (!currentBlob) break;

            // Start pre-fetching the next chunk while we prepare to play this one
            let prefetchPromise = (i + 1 < chunks.length) ? fetchAudio(chunks[i+1]) : null;

            // Convert blob to URL and play
            const url = URL.createObjectURL(currentBlob);
            currentTTS.audio = new Audio(url);

            try {
                await new Promise((resolve, reject) => {
                    currentTTS.audio.onplay = () => {
                        // Once the first chunk successfully starts, iOS considers the "session" active
                        if (prefetchPromise) prefetchPromise.then(b => nextBlob = b);
                    };
                    currentTTS.audio.onended = resolve;
                    currentTTS.audio.onerror = reject;
                    currentTTS.audio.play().catch(reject);
                });
            } catch(e) {
                console.error("Playback error", e);
                // If it fails here, it's likely still an autoplay block
                if (e.name === "NotAllowedError") {
                    alert("iOS blocked playback. Try tapping 'Narrate' again.");
                }
                break;
            }

            URL.revokeObjectURL(url);
            if (prefetchPromise) nextBlob = await prefetchPromise;
        }

        if (currentTTS.msgIndex === index) {
            currentTTS.msgIndex = -1;
            if (btn) btn.textContent = '▶ Narrate';
        }
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

    async function handleAIRefine(index) {
        if (!confirm("Have Venice rewrite this message to remove banned phrases?")) return;

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
                body: JSON.stringify({index: index})
            });
            const d = await res.json();
            if (d.success) {
                chatHistory[index].original_content = originalText;
                chatHistory[index].content = d.refined_text;
                diffModeMsgs.add(index);
                renderChat();
            } else {
                alert("Refinement Error: " + d.error);
                if (overlay) overlay.remove();
            }
        } catch(e) {
            alert("Failed to refine message.");
            if (overlay) overlay.remove();
        }
    }

    async function loadHistory() {
        try {
            const res = await fetch('/get_history');
            const data = await res.json();
            chatHistory = data.history || [];
            summaries = data.summaries || [];
            currentVisualMemory = data.visual_memory || "";
            if (vmTextarea) vmTextarea.value = currentVisualMemory;

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

    function renderChat() {
        if (!chatbox) return;
        chatbox.innerHTML = '';
        runningInput = 0;
        runningOutput = 0;
        let textMsgCounter = 0;

        let lastSumIndex = -1;
        if (summaries && summaries.length > 0) {
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

            const isSummed = (idx <= lastSumIndex);
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

                    const contentDiv = document.createElement('div');
                    contentDiv.style.cssText = "white-space:pre-wrap; font-family:inherit;";

                    if (diffModeMsgs.has(idx) && msg.original_content) {
                        const origText = Array.isArray(msg.original_content) ? msg.original_content.find(p=>p.type==='text')?.text || "" : msg.original_content;
                        const curText = Array.isArray(msg.content) ? msg.content.find(p=>p.type==='text')?.text || "" : msg.content;
                        contentDiv.innerHTML = renderDiff(origText, curText);
                    } else {
                        let textContent = Array.isArray(msg.content) ? (msg.content.find(p => p.type === 'text')?.text || "") : msg.content;
                        contentDiv.innerHTML = typeof marked !== 'undefined' ? marked.parse(textContent || "") : textContent;

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
            optionsBtn.textContent = '⋮';

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

            const refineBtn = document.createElement('button'); refineBtn.className = 'msg-btn'; refineBtn.textContent = 'AI Refine';
            refineBtn.onclick = () => handleAIRefine(idx);

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
                ttsBtn.style.borderColor = 'var(--accent)';
                ttsBtn.style.color = 'var(--accent)';
                ttsBtn.textContent = '▶ Narrate';
                ttsBtn.onclick = () => handleTTS(idx);
                btnRow.append(ttsBtn);
            }

            btnRow.append(branchBtn, refineBtn);
            toolsPanel.appendChild(btnRow);

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
                            statHtml += `<br><span style="font-size:0.85em; color:var(--text-dim);">├ Sum: ${u.breakdown.summary || 0}</span>`;
                            statHtml += `<br><span style="font-size:0.85em; color:var(--text-dim);">├ Raw: ${u.breakdown.raw || 0}</span>`;
                            statHtml += `<br><span style="font-size:0.85em; color:var(--text-dim);">└ Sys: ${u.breakdown.system || 0}</span>`;
                        }

                        let cachedTokens = 0;
                        if (u.prompt_tokens_details && u.prompt_tokens_details.cached_tokens) {
                            cachedTokens = u.prompt_tokens_details.cached_tokens;
                        }
                        if (cachedTokens > 0) {
                            statHtml += `<br><span style="font-size:0.85em; color:#10b981;">Cached: ${cachedTokens} (${Math.round((cachedTokens/u.prompt_tokens)*100)}%)</span>`;
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
                if (myBatch) {
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
            chatbox.appendChild(div);
        });
        chatbox.scrollTop = chatbox.scrollHeight;
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
            title.textContent = `Batch #${i + 1} (Msgs #${startNum} - #${endNum})`;
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
            if (userInput) userInput.value = txt;
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

        let messagePayload = txt;
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
        triggerGen(messagePayload, null, sendTime);
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
            }

            if (finalUsageA) { compUsageA.prompt += finalUsageA.prompt_tokens; compUsageA.comp += finalUsageA.completion_tokens; }
            if (finalUsageB) { compUsageB.prompt += finalUsageB.prompt_tokens; compUsageB.comp += finalUsageB.completion_tokens; }

            const aEntry = {role: 'assistant', content: fullA, timestamp: new Date().toISOString()};
            if (finalUsageA) aEntry.usage = finalUsageA;
            compareHistoryA.push(aEntry);

            const bEntry = {role: 'assistant', content: fullB, timestamp: new Date().toISOString()};
            if (finalUsageB) bEntry.usage = finalUsageB;
            compareHistoryB.push(bEntry);

            const renderUsage = (usage, compUsage) => {
                let html = `Session Tokens: <span style="color:inherit">${compUsage.prompt + compUsage.comp}</span> (In: ${compUsage.prompt} | Out: ${compUsage.comp})`;
                if (usage && usage.prompt_tokens_details && usage.prompt_tokens_details.cached_tokens) {
                    html += `<br><span style="color:#10b981;">Latest request cached: ${usage.prompt_tokens_details.cached_tokens}</span>`;
                }
                return html;
            };

            if (colA_stats) colA_stats.innerHTML = renderUsage(finalUsageA, compUsageA);
            if (colB_stats) colB_stats.innerHTML = renderUsage(finalUsageB, compUsageB);

            compareContainerEl.querySelector('#keep-btn-a').style.display = 'block';
            compareContainerEl.querySelector('#keep-btn-b').style.display = 'block';

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
            const res = await fetch('/venice_models');
            const mData = await res.json();
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

    async function triggerGen(msg, customModel = null, timestamp = null) {
        lastWfmText = "";
        lastWfmUsage = null;
        isGenerating = true;
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
            const res = await fetch('/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
            const reader = res.body.getReader();
            const dec = new TextDecoder();

            let rBlock = null;
            let contentDiv = null;

            while(true){
                const {done, value} = await reader.read();
                if(done) break;
                const chunk = dec.decode(value);
                chunk.split('\n\n').forEach(l=>{
                   if(l.startsWith('data: ')){
                       try{ 
                           const d = JSON.parse(l.substring(6)); 

                           if(d.reasoning) {
                               if(!rBlock) {
                                   div.innerHTML = '';
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
                           }

                           if(d.content) {
                               if(!contentDiv) {
                                   contentDiv = document.createElement('div');
                                   contentDiv.style.cssText = "white-space:pre-wrap; font-family:inherit;";
                                   div.appendChild(contentDiv);
                                   // Auto-collapse reasoning when content starts
                                   if(rBlock) {
                                       rBlock.querySelector('.reasoning-content').style.display = 'none';
                                       rBlock.querySelector('.toggle-icon').textContent = '▼';
                                       rBlock.querySelector('.reasoning-header').onclick = () => {
                                           const c = rBlock.querySelector('.reasoning-content');
                                           const i = rBlock.querySelector('.toggle-icon');
                                           const isH = c.style.display === 'none';
                                           c.style.display = isH ? 'block' : 'none';
                                           i.textContent = isH ? '▲' : '▼';
                                       };
                                   }
                               }
                               full += d.content;
                               contentDiv.innerHTML = typeof marked !== 'undefined' ? marked.parse(full) : full;
                           }

                           if(d.balance) updateBalanceDisplay(d.balance);
                       }catch(e){}
                   } 
                });
                if (chatbox) chatbox.scrollTop = chatbox.scrollHeight;
            }
            await loadHistory();
        } catch(e) { div.textContent = "Error"; }
        finally { isGenerating = false; }
    }

    async function handleImageBtn() {
        if(guidedModeEnabled) { 
            if (guidedOverlay) guidedOverlay.style.display='block'; 
            if (guidedInput) guidedInput.focus(); 
        }
        else triggerImageGen("");
    }

    async function triggerImageGen(g, debugMode = false) {
        if(isGenerating) return; 
        isGenerating = true; 
        if (guidedOverlay) guidedOverlay.style.display='none';
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
            data.chats.forEach(file => {
                const li = document.createElement('li'); li.className = 'chat-item';
                if (file === data.active_chat) li.classList.add('active-chat');
                li.innerHTML = `<span class="chat-name">${file.replace('.json','').replace(/_/g,' ')}</span>
                                <div><button class="icon-btn r-btn">✎</button><button class="icon-btn d-btn">🗑</button></div>`;
                li.onclick = (e) => { if(!e.target.closest('.icon-btn')) loadChat(file); };
                li.querySelector('.r-btn').onclick = () => renameChat(file);
                li.querySelector('.d-btn').onclick = () => deleteChat(file);
                list.appendChild(li);
            });
        } catch(e) {
            console.error("Sidebar load failed:", e);
        }
    }

    window.renameChat = async (old) => {
        const n = prompt("Rename:", old.replace('.json',''));
        if(n) { await fetch('/rename_chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({old:old, new:n})}); loadSidebar(); }
    };
    window.deleteChat = async (file) => {
        if(confirm("Delete?")) { await fetch('/delete_chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({filename:file})}); loadSidebar(); }
    };
    async function loadChat(file) {
        diffModeMsgs.clear();
        isComparing = false;
        await fetch('/load_chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({filename:file})});
        loadHistory(); 
        const sidebar = document.getElementById('sidebar');
        if (sidebar) sidebar.classList.add('hidden');
    }

    const openSettings = async () => {
        try {
            const res = await fetch('/get_settings');
            const d = await res.json();

            const mRes = await fetch('/venice_models');
            const mData = await mRes.json();

            const populateModelSelect = (id, currentVal) => {
                const select = document.getElementById(id);
                if (!select) return;
                select.innerHTML = '';
                for (const group in mData) {
                    const optgroup = document.createElement('optgroup');
                    optgroup.label = group;
                    mData[group].forEach(m => {
                        const opt = document.createElement('option');
                        opt.value = m.id;
                        // Add Eye emoji to vision models
                        opt.textContent = (m.vision ? '👁️ ' : '') + m.name;
                        if (m.id === currentVal) opt.selected = true;
                        optgroup.appendChild(opt);
                    });
                    select.appendChild(optgroup);
                }
            };

            const bRes = await fetch('/get_banned_phrases');
            const bData = await bRes.json();
            const bannedTa = document.getElementById('set-banned-phrases');
            if (bannedTa) bannedTa.value = bData.banned_phrases || "";

            const dl = document.getElementById('model-history-list'); 
            if (dl) {
                dl.innerHTML = '';
                d.model_history.forEach(m => { const o = document.createElement('option'); o.value = m; dl.appendChild(o); });
            }

            const setVal = (id, val) => { const el = document.getElementById(id); if (el) el.value = val; };
            const setCheck = (id, checked) => { const el = document.getElementById(id); if (el) el.checked = checked; };

            populateModelSelect('set-model-chat', d.venice.model);
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
                    const isReasoning = modelId.includes('kimi-k2') || modelId.toLowerCase().includes('deepseek');
                    reasoningContainer.style.display = isReasoning ? 'block' : 'none';
                }
            };

            if (modelSelect) {
                modelSelect.onchange = (e) => updateModelInfo(e.target.value);
                updateModelInfo(modelSelect.value);
            }

            populateModelSelect('set-model-wfm', d.wfm.model || "venice-uncensored");
            setVal('set-temp-wfm', d.wfm.temperature || 0.8);
            setVal('set-depth-wfm', d.wfm.context_depth || 10);

            populateModelSelect('set-model-img', d.venice_img.model || "qwen3-4b");
            setVal('set-gen-model-img', d.image_gen.model || "lustify-v7");
            setVal('set-img-steps', d.image_gen.steps || 40);
            setVal('set-img-cfg', d.image_gen.cfg_scale || 7.5);

            const imgModelSelect = document.getElementById('set-gen-model-img');
            const updateRecs = (modelId) => {
                const defs = {
                    "lustify-v7": { steps: 40, cfg: 7.5 },
                    "lustify-sdxl": { steps: 40, cfg: 7.5 },
                    "anime-wai": { steps: 40, cfg: 7.5 },
                    "chroma": { steps: 40, cfg: 7.5 },
                    "hidream": { steps: 40, cfg: 7.5 },
                    "z-image-turbo": { steps: 6, cfg: 1.5 },
                    "qwen-image": { steps: 6, cfg: 1.5 },
                    "qwen-image-2": { steps: 8, cfg: 2.0 },
                    "flux-2-pro": { steps: 25, cfg: 3.5 },
                    "grok-imagine": { steps: 25, cfg: 3.5 },
                    "venice-sd35": { steps: 30, cfg: 5.0 },
                    "recraft-v4": { steps: 30, cfg: 6.0 },
                    "seeddream-v4.5": { steps: 30, cfg: 5.0 },
                    "seeddream-v5-lite": { steps: 30, cfg: 5.0 },
                    "imagineart-1.5-pro": { steps: 30, cfg: 5.0 }
                };
                const rec = defs[modelId] || { steps: 30, cfg: 7.0 };
                const sEl = document.getElementById('rec-steps');
                const cEl = document.getElementById('rec-cfg');
                if (sEl) sEl.textContent = `(Rec: ${rec.steps})`;
                if (cEl) cEl.textContent = `(Rec: ${rec.cfg})`;
            };
            if (imgModelSelect) {
                imgModelSelect.onchange = (e) => updateRecs(e.target.value);
                updateRecs(imgModelSelect.value);
            }

            setVal('set-art-style', d.image_gen.art_style || "None");
            setVal('set-negative-styles', d.image_gen.negative_styles || "");
            setVal('set-img-depth', d.venice_img.context_depth || 3);

            const fontSize = d.interface?.font_size || 16;
            const bgColor = d.interface?.bg_color || "#121212";
            setVal('set-font-size', fontSize);
            setVal('set-bg-color', bgColor);
            applyInterfaceSettings(fontSize, bgColor);

            const w = d.image_gen.width || 1024; const h = d.image_gen.height || 1024;
            setCheck('set-hd', (w > 1300 || h > 1300));
            guidedModeEnabled = d.image_gen.guided_mode || false;
            setCheck('set-guided', guidedModeEnabled);

            setCheck('sum-enabled', d.summarizer.enabled);
            populateModelSelect('sum-model', d.summarizer.model || "qwen3-4b");
            setVal('sum-threshold', d.summarizer.trigger_threshold_turns);
            setVal('sum-batch', d.summarizer.batch_size);
            setVal('sum-keep', d.summarizer.recent_turns_to_keep);
            setVal('sum-prompt', d.summarizer.system_prompt);
            populateModelSelect('sum-cons-model', d.summarizer.consolidation_model || "venice-uncensored");
            setVal('sum-cons-prompt', d.summarizer.consolidation_prompt || "Summarize.");

            const sumOpts = document.getElementById('sum-options');
            if (sumOpts) sumOpts.style.display = d.summarizer.enabled ? 'block' : 'none';

            setCheck('rag-enabled', d.rag.enabled);
            setVal('rag-k', d.rag.k);
            setVal('rag-max-chars', d.rag.max_chars);
            setVal('rag-min-chars', d.rag.min_chars);

            const ragOpts = document.getElementById('rag-options');
            if (ragOpts) ragOpts.style.display = d.rag.enabled ? 'block' : 'none';

            setCheck('tts-enabled', d.tts.enabled);
            setVal('set-tts-model', d.tts.model || "tts-kokoro");
            setVal('set-tts-voice', d.tts.voice || "af_sky");
            setVal('set-tts-speed', d.tts.speed || 1.0);
            const speedDisp = document.getElementById('tts-speed-val');
            if(speedDisp) speedDisp.textContent = (d.tts.speed || 1.0) + "x";

            const ttsOpts = document.getElementById('tts-options');
            if (ttsOpts) ttsOpts.style.display = d.tts.enabled ? 'block' : 'none';

            if (modal) modal.style.display = 'block';
        } catch(e) {
            console.error("Failed to open settings:", e);
        }
    }

    function applyInterfaceSettings(fontSize, bgColor) {
        document.documentElement.style.setProperty('--font-size', fontSize + 'px');
        document.documentElement.style.setProperty('--bg-dark', bgColor);
    }

    function calcPixels(ar, hd) {
        let w=1024, h=1024;

        // Use Venice-optimized "Golden Ratio" multiples of 32
        if(ar==="4:3") { w=1152; h=864; } 
        else if(ar==="3:4") { w=864; h=1152; }
        else if(ar==="16:9") { w=1216; h=704; } 
        else if(ar==="9:16") { w=704; h=1216; }
        else if(ar==="2:3") { w=832; h=1248; }
        else if(ar==="3:2") { w=1248; h=832; }

        if(hd) { 
            // 1.2x boost for HD mode while staying within stability limits
            w = Math.round((w * 1.2) / 32) * 32; 
            h = Math.round((h * 1.2) / 32) * 32; 
        }

        // Strict Caps for stability on decentralized nodes
        // Hard cap at 1280 for standard models to prevent black images (NaN errors)
        w = Math.min(w, 1280);
        h = Math.min(h, 1280);

        return {w, h};
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

    safeBind('sendBtn', 'click', sendMessage);
    safeBind('genImgBtn', 'click', handleImageBtn);
    safeBind('toggleGuidedBtn', 'click', () => {
        if (!guidedOverlay) return;
        guidedOverlay.style.display = (guidedOverlay.style.display === 'none') ? 'block' : 'none';
        if (guidedOverlay.style.display === 'block' && guidedInput) guidedInput.focus();
    });
    safeBind('guided-submit', 'click', () => {
        const debugMode = document.getElementById('guided-debug-mode')?.checked || false;
        triggerImageGen(guidedInput?.value, debugMode);
    });
    safeBind('guided-cancel', 'click', () => { if(guidedOverlay) guidedOverlay.style.display='none'; });

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

    safeBind('auto-rename-btn', 'click', async () => {
        const btn = document.getElementById('auto-rename-btn');
        const modelSelect = document.getElementById('rename-model-select');
        if (!btn || !modelSelect) return;

        const ogText = btn.textContent;
        btn.textContent = "⌛"; btn.disabled = true;

        try {
            const res = await fetch('/generate_chat_title', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ model: modelSelect.value })
            });
            const d = await res.json();

            if (d.success) {
                await fetch('/rename_chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ old: d.old_filename, new: d.title })
                });
                await loadSidebar();
                // We don't need to reload history as the content hasn't changed, 
                // but the sidebar needs to reflect the new name.
            } else {
                alert("Rename Error: " + d.error);
            }
        } catch(e) {
            console.error(e);
            alert("Failed to auto-rename.");
        }
        btn.textContent = ogText; btn.disabled = false;
    });

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

    safeBind('scan-visuals-btn', 'click', async () => {
        const btn = document.getElementById('scan-visuals-btn');
        const depthEl = document.getElementById('set-visual-depth');
        const depthVal = depthEl ? depthEl.value : 50;
        if (btn) { btn.textContent = "Scanning..."; btn.disabled = true; }
        try {
            const res = await fetch('/scan_visuals', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({depth: parseInt(depthVal)}) });
            const d = await res.json();
            if(d.success) { 
                if (vmTextarea) vmTextarea.value = d.memory; 
                currentVisualMemory = d.memory; 
                await loadHistory(); 
            }
        } catch(e) { alert("Scan Error"); }
        finally { if (btn) { btn.textContent = "Scan Context"; btn.disabled = false; } }
    });

    safeBind('save-visuals-btn', 'click', async () => {
        const val = vmTextarea ? vmTextarea.value : "";
        await fetch('/update_visual_memory', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({memory: val}) });
        alert("Memory Saved"); await loadHistory();
    });

    safeBind('sum-enabled', 'click', async (e) => {
        const isChecked = e.target.checked;
        const opts = document.getElementById('sum-options');
        if (opts) opts.style.display = isChecked ? 'block' : 'none';

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
        const hdEl = document.getElementById('set-hd');
        const guidedEl = document.getElementById('set-guided');

        const ar = arEl ? arEl.value : "3:4";
        const hd = hdEl ? hdEl.checked : true;
        const {w,h} = calcPixels(ar,hd);
        guidedModeEnabled = guidedEl ? guidedEl.checked : false;

        const getVal = (id, def) => { const el = document.getElementById(id); return el ? el.value : def; };
        const getCheck = (id, def) => { const el = document.getElementById(id); return el ? el.checked : def; };

        const fontSize = getVal('set-font-size', '16');
        const bgColor = getVal('set-bg-color', '#121212');

        const payload = {
            venice: { 
                model: getVal('set-model-chat', 'venice-uncensored'), 
                include_venice_system_prompt: getCheck('set-venice-system', true),
                vision_high_res: getCheck('set-vision-high-res', true),
                temperature: parseFloat(getVal('set-temp-chat', 0.8)),
                max_tokens: parseInt(getVal('set-tokens-chat', 4000)),
                reasoning_effort: getVal('set-reasoning-effort', 'medium'),
                frequency_penalty: parseFloat(getVal('set-freq-chat', 0)),
                presence_penalty: parseFloat(getVal('set-pres-chat', 0))
            },
            wfm: {
                model: getVal('set-model-wfm', 'venice-uncensored'),
                temperature: parseFloat(getVal('set-temp-wfm', 0.8)),
                context_depth: parseInt(getVal('set-depth-wfm', 10))
            },
            venice_img: { model: getVal('set-model-img', 'qwen3-4b'), context_depth: parseInt(getVal('set-img-depth', 3)) },
            summarizer: { 
                enabled: getCheck('sum-enabled', false), 
                model: getVal('sum-model', 'qwen3-4b'),
                trigger_threshold_turns: parseInt(getVal('sum-threshold', 12)), 
                batch_size: parseInt(getVal('sum-batch', 4)), 
                recent_turns_to_keep: parseInt(getVal('sum-keep', 12)), 
                system_prompt: getVal('sum-prompt', 'Summarize.'),
                consolidation_model: getVal('sum-cons-model', 'venice-uncensored'),
                consolidation_prompt: getVal('sum-cons-prompt', 'Summarize.')
            },
            rag: {
                enabled: getCheck('rag-enabled', false),
                k: parseInt(getVal('rag-k', 3)),
                max_chars: parseInt(getVal('rag-max-chars', 1200)),
                min_chars: parseInt(getVal('rag-min-chars', 200))
            },
            tts: {
                enabled: getCheck('tts-enabled', false),
                model: getVal('set-tts-model', 'tts-kokoro'),
                voice: getVal('set-tts-voice', 'af_sky'),
                speed: parseFloat(getVal('set-tts-speed', 1.0))
            },
            image_gen: { 
                model: getVal('set-gen-model-img', 'lustify-v7'),
                steps: parseInt(getVal('set-img-steps', 40)),
                cfg_scale: parseFloat(getVal('set-img-cfg', 7.5)),
                width:w, height:h, 
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

        const ogText = btn.textContent;
        btn.textContent = "⏳"; btn.disabled = true;
        try {
            const res = await fetch('/write_for_me', { method:'POST' });
            const d = await res.json();
            if(d.success) { 
                if (userInput) {
                    userInput.value = d.text; 
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
            setTxt('stat-p-sys', gData.proj_sys.toLocaleString());
        } catch (e) {
            console.error("Failed to load global stats", e);
        }

        if (analyticsModal) analyticsModal.style.display = 'block';
    });

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
        if (modal) modal.style.display = 'block';
        await loadCacheDebug();
    });

    safeBind('close-cache-debug-x', 'click', () => {
        const modal = document.getElementById('cache-debug-modal');
        if (modal) modal.style.display = 'none';
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
});