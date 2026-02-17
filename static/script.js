// Last Updated: 2026-02-17 13:00:00
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

    const chatbox = document.getElementById('chatbox');
    const userInput = document.getElementById('userInput');
    const modal = document.getElementById('settingsModal');
    const editOverlay = document.getElementById('edit-overlay');
    const editTextarea = document.getElementById('edit-textarea');
    const guidedOverlay = document.getElementById('guided-overlay');
    const architectModal = document.getElementById('architectModal');
    const architectChatBox = document.getElementById('architect-chat');
    const architectInput = document.getElementById('architect-input');
    let architectHistory = [];

    loadHistory();
    loadSidebar();

    // ... (Existing loadHistory and renderChat functions) ...

    async function loadHistory() {
        try {
            const res = await fetch('/get_history');
            const data = await res.json();
            chatHistory = data.history || [];
            summaries = data.summaries || [];
            currentVisualMemory = data.visual_memory || "";
            if (vmTextarea) vmTextarea.value = currentVisualMemory;
            renderChat();
        } catch (e) { console.error("History error", e); }
    }

    // ... (rest of code) ...

    document.getElementById('architect-btn').onclick = () => {
        architectModal.style.display = 'block';
        architectHistory = []; // Reset history
        architectChatBox.innerHTML = '<div class="message assistant">Hello! I\'m the Architect. I can help you design a new scenario, RPG, or story. What kind of experience are you looking for today?</div>';
    };

    document.getElementById('close-architect').onclick = () => architectModal.style.display = 'none';

    document.getElementById('architect-send').onclick = sendArchitectMessage;
    architectInput.onkeydown = (e) => { if(e.key==='Enter') sendArchitectMessage(); };

    async function sendArchitectMessage() {
        const txt = architectInput.value.trim();
        if(!txt) return;

        // Add User Message
        const uDiv = document.createElement('div'); uDiv.className = 'message user'; uDiv.textContent = txt;
        architectChatBox.appendChild(uDiv);
        architectInput.value = '';

        // Add Assistant Placeholder
        const aDiv = document.createElement('div'); aDiv.className = 'message assistant'; aDiv.textContent = '...';
        architectChatBox.appendChild(aDiv);
        architectChatBox.scrollTop = architectChatBox.scrollHeight;

        let fullContent = "";
        let isScenarioReady = false;
        let ghostPrompt = "";

        try {
            const res = await fetch('/architect_chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ history: architectHistory, message: txt })
            });

            const reader = res.body.getReader();
            const dec = new TextDecoder();

            while(true) {
                const {done, value} = await reader.read();
                if(done) break;
                const chunk = dec.decode(value);
                const lines = chunk.split('\n\n');
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const d = JSON.parse(line.substring(6));
                            if (d.content) {
                                fullContent += d.content;

                                // Check for Magic Token
                                if (fullContent.includes('__SCENARIO_READY__')) {
                                    isScenarioReady = true;
                                    const parts = fullContent.split('__SCENARIO_READY__');
                                    // We show the part BEFORE the token
                                    aDiv.innerHTML = marked.parse(parts[0]);
                                    // We capture the part AFTER the token
                                    ghostPrompt = parts[1];
                                } else {
                                    if (!isScenarioReady) {
                                        aDiv.innerHTML = marked.parse(fullContent);
                                    } else {
                                        // If ready, we keep appending to ghostPrompt invisible to user
                                        // (Actually fullContent has it, so we just parse at the end)
                                    }
                                }
                                architectChatBox.scrollTop = architectChatBox.scrollHeight;
                            }
                        } catch(e) {}
                    }
                }
            }

            // Round finished
            if (isScenarioReady) {
                // Final parse to ensure we got everything
                const parts = fullContent.split('__SCENARIO_READY__');
                ghostPrompt = parts[1].trim();

                // Trigger the handoff
                const launchBtn = document.createElement('button');
                launchBtn.className = 'action';
                launchBtn.style.width = '100%';
                launchBtn.style.marginTop = '10px';
                launchBtn.style.background = 'var(--accent)';
                launchBtn.textContent = "🚀 Launch Scenario";
                launchBtn.onclick = () => launchScenario(ghostPrompt);

                const readyMsg = document.createElement('div');
                readyMsg.className = 'message assistant';
                readyMsg.innerHTML = "<b>Scenario Generated!</b> Click below to start.";
                readyMsg.appendChild(launchBtn);
                architectChatBox.appendChild(readyMsg);
                architectChatBox.scrollTop = architectChatBox.scrollHeight;

            } else {
                architectHistory.push({role: 'user', content: txt});
                architectHistory.push({role: 'assistant', content: fullContent});
            }

        } catch(e) { aDiv.textContent = "Error: " + e; }
    }

    async function launchScenario(promptText) {
        try {
            const res = await fetch('/create_scenario_chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ prompt: promptText })
            });
            const d = await res.json();
            if(d.success) {
                architectModal.style.display = 'none';
                loadHistory(); // Reloads into the new chat
                document.getElementById('sidebar').classList.add('hidden');
            }
        } catch(e) { alert("Launch failed: " + e); }
    }

    document.getElementById('save-large-edit').onclick = async () => {
        const val = editTextarea.value;
        if (activeEditType === 'message') {
            chatHistory[activeEditIndex].content = val;
            await fetch('/update_history', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({history:chatHistory}) });
            renderChat();
        } else {
            await fetch('/save_settings', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({main_prompt:val}) });
        }
        editOverlay.style.display = 'none';
    };
    document.getElementById('cancel-large-edit').onclick = () => editOverlay.style.display = 'none';

    async function handleRegen(idx, role) {
        if (!confirm("Regenerate? Future messages will be deleted.")) return;
        if (role === 'user') {
            const txt = chatHistory[idx].content;
            chatHistory = chatHistory.slice(0, idx);
            userInput.value = txt;
            await saveHistory(); renderChat();
        } else {
            chatHistory = chatHistory.slice(0, idx);
            await saveHistory(); renderChat(); triggerGen("");
        }
    }

    async function saveHistory() {
        await fetch('/update_history', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({history:chatHistory}) });
    }

    async function sendMessage() {
        const txt = userInput.value.trim();
        if(!txt || isGenerating) return;
        userInput.value = '';
        chatHistory.push({role:'user', content:txt});
        renderChat();
        triggerGen(txt);
    }

    async function triggerGen(msg) {
        isGenerating = true;
        const div = document.createElement('div'); div.className = 'message assistant'; div.textContent = '...';
        chatbox.appendChild(div); chatbox.scrollTop = chatbox.scrollHeight;

        let full = "";
        try {
            const res = await fetch('/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({message:msg}) });
            const reader = res.body.getReader();
            const dec = new TextDecoder();
            while(true){
                const {done, value} = await reader.read();
                if(done) break;
                const chunk = dec.decode(value);
                chunk.split('\n\n').forEach(l=>{
                   if(l.startsWith('data: ')){
                       try{ const d = JSON.parse(l.substring(6)); if(d.content) full+=d.content; }catch(e){}
                   } 
                });
                div.innerHTML = marked.parse(full);
                chatbox.scrollTop = chatbox.scrollHeight;
            }
            await loadHistory();
        } catch(e) { div.textContent = "Error"; }
        finally { isGenerating = false; }
    }

    async function handleImageBtn() {
        if(guidedModeEnabled) { guidedOverlay.style.display='block'; guidedInput.focus(); }
        else triggerImageGen("");
    }

    async function triggerImageGen(g) {
        if(isGenerating) return; 
        isGenerating = true; 
        guidedOverlay.style.display='none';

        const genBtn = document.getElementById('genImgBtn');
        const originalBtnText = genBtn.textContent;
        genBtn.textContent = "Generating Image...";
        genBtn.disabled = true;

        // Add a temporary loading message to the chat
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message assistant system-msg';
        loadingDiv.id = 'temp-image-loading';
        loadingDiv.innerHTML = `<i>Painting image... please wait.</i>`;
        chatbox.appendChild(loadingDiv);
        chatbox.scrollTop = chatbox.scrollHeight;

        try {
            const res = await fetch('/generate_image', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({guidance:g})});
            if(res.ok) {
                await loadHistory(); 
            } else {
                const errData = await res.json();
                alert("Gen Failed: " + (errData.error || "Unknown Error"));
            }
        } catch(e) { 
            alert("API Error: " + e.message); 
        } finally { 
            isGenerating = false; 
            genBtn.textContent = originalBtnText;
            genBtn.disabled = false;
            const tempMsg = document.getElementById('temp-image-loading');
            if(tempMsg) tempMsg.remove();
        }
    }

    document.getElementById('scan-visuals-btn').onclick = async () => {
        const btn = document.getElementById('scan-visuals-btn');
        const depthVal = document.getElementById('set-visual-depth').value || 50;
        btn.textContent = "Scanning..."; btn.disabled = true;
        try {
            const res = await fetch('/scan_visuals', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({depth: parseInt(depthVal)}) });
            const d = await res.json();
            if(d.success) { vmTextarea.value = d.memory; currentVisualMemory = d.memory; await loadHistory(); }
        } catch(e) { alert("Scan Error"); }
        finally { btn.textContent = "Scan Context"; btn.disabled = false; }
    };

    document.getElementById('save-visuals-btn').onclick = async () => {
        await fetch('/update_visual_memory', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({memory: vmTextarea.value}) });
        alert("Memory Saved"); await loadHistory();
    };

    async function loadSidebar() {
        const res = await fetch('/sidebar_data');
        const data = await res.json();
        const list = document.getElementById('chat-list');
        list.innerHTML = '';
        data.chats.forEach(file => {
            const li = document.createElement('li'); li.className = 'chat-item';
            li.innerHTML = `<span class="chat-name">${file.replace('.json','').replace(/_/g,' ')}</span>
                            <div><button class="icon-btn r-btn">✎</button><button class="icon-btn d-btn">🗑</button></div>`;
            li.onclick = (e) => { if(!e.target.closest('.icon-btn')) loadChat(file); };
            li.querySelector('.r-btn').onclick = () => renameChat(file);
            li.querySelector('.d-btn').onclick = () => deleteChat(file);
            list.appendChild(li);
        });
    }

    window.renameChat = async (old) => {
        const n = prompt("Rename:", old.replace('.json',''));
        if(n) { await fetch('/rename_chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({old:old, new:n})}); loadSidebar(); }
    };
    window.deleteChat = async (file) => {
        if(confirm("Delete?")) { await fetch('/delete_chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({filename:file})}); loadSidebar(); }
    };
    async function loadChat(file) {
        await fetch('/load_chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({filename:file})});
        loadHistory(); document.getElementById('sidebar').classList.add('hidden');
    }

    async function openSettings() {
        const res = await fetch('/get_settings');
        const d = await res.json();

        const dl = document.getElementById('model-history-list'); dl.innerHTML = '';
        d.model_history.forEach(m => { const o = document.createElement('option'); o.value = m; dl.appendChild(o); });

        document.getElementById('set-model-chat').value = d.mistral.model;
        document.getElementById('set-temp-chat').value = d.mistral.temperature;
        document.getElementById('set-tokens-chat').value = d.mistral.max_tokens;
        document.getElementById('set-freq-chat').value = d.mistral.frequency_penalty;
        document.getElementById('set-pres-chat').value = d.mistral.presence_penalty;

        document.getElementById('set-model-img').value = d.mistral_img.model || "mistral-small-latest";
        document.getElementById('set-img-depth').value = d.mistral_img.context_depth || 3;

        const w = d.fal.width || 1024;
        const h = d.fal.height || 1024;
        const hd = (w > 1300 || h > 1300);
        document.getElementById('set-hd').checked = hd;
        guidedModeEnabled = d.fal.guided_mode || false;
        document.getElementById('set-guided').checked = guidedModeEnabled;

        // Summarizer
        document.getElementById('sum-enabled').checked = d.summarizer.enabled;
        document.getElementById('sum-threshold').value = d.summarizer.trigger_threshold_turns;
        document.getElementById('sum-batch').value = d.summarizer.batch_size;
        document.getElementById('sum-keep').value = d.summarizer.recent_turns_to_keep;
        document.getElementById('sum-prompt').value = d.summarizer.system_prompt;
        document.getElementById('sum-options').style.display = d.summarizer.enabled ? 'block' : 'none';

        modal.style.display = 'block';
    }

    // Logic for "Enabling Late" Confirmation
    document.getElementById('sum-enabled').onclick = async (e) => {
        const isChecked = e.target.checked;
        document.getElementById('sum-options').style.display = isChecked ? 'block' : 'none';

        if (isChecked) {
            // Check status via API
            const tempSettings = {
                recent_turns_to_keep: document.getElementById('sum-keep').value,
                batch_size: document.getElementById('sum-batch').value
            };

            try {
                const res = await fetch('/check_summary_status', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({settings: tempSettings})
                });
                const stats = await res.json();

                if (stats.batches_pending > 0) {
                    const confirmMsg = `Enabling summarization now will process ${stats.unsummarized_count} messages into ${stats.batches_pending} separate batches. Proceed?`;
                    if (!confirm(confirmMsg)) {
                        e.target.checked = false;
                        document.getElementById('sum-options').style.display = 'none';
                    }
                }
            } catch(err) { console.error(err); }
        }
    };

    function calcPixels(ar, hd) {
        let w=1024, h=1024;
        if(ar==="1:1"){w=1024;h=1024;} else if(ar==="4:3"){w=1152;h=864;} else if(ar==="3:4"){w=864;h=1152;}
        else if(ar==="16:9"){w=1344;h=768;} else if(ar==="9:16"){w=768;h=1344;}
        if(hd){ w=Math.round((w*1.5)/32)*32; h=Math.round((h*1.5)/32)*32; }
        return {w,h};
    }

    document.getElementById('save-settings-btn').onclick = async () => {
        const ar = document.getElementById('set-ar').value;
        const hd = document.getElementById('set-hd').checked;
        const {w,h} = calcPixels(ar,hd);
        guidedModeEnabled = document.getElementById('set-guided').checked;

        const payload = {
            mistral: { 
                model: document.getElementById('set-model-chat').value, 
                temperature: parseFloat(document.getElementById('set-temp-chat').value),
                max_tokens: parseInt(document.getElementById('set-tokens-chat').value),
                frequency_penalty: parseFloat(document.getElementById('set-freq-chat').value),
                presence_penalty: parseFloat(document.getElementById('set-pres-chat').value)
            },
            mistral_img: {
                model: document.getElementById('set-model-img').value,
                context_depth: parseInt(document.getElementById('set-img-depth').value)
            },
            summarizer: {
                enabled: document.getElementById('sum-enabled').checked,
                trigger_threshold_turns: parseInt(document.getElementById('sum-threshold').value),
                batch_size: parseInt(document.getElementById('sum-batch').value),
                recent_turns_to_keep: parseInt(document.getElementById('sum-keep').value),
                system_prompt: document.getElementById('sum-prompt').value
            },
            fal: { width:w, height:h, guided_mode: guidedModeEnabled }
        };
        await fetch('/save_settings', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
        modal.style.display = 'none';
        loadHistory();
    };

    document.getElementById('sendBtn').onclick = sendMessage;
    document.getElementById('genImgBtn').onclick = handleImageBtn;
    document.getElementById('guided-submit').onclick = () => triggerImageGen(guidedInput.value);
    document.getElementById('guided-cancel').onclick = () => guidedOverlay.style.display='none';
    document.getElementById('menuBtn').onclick = openSettings;
    document.getElementById('mobile-menu-btn').onclick = () => { loadSidebar(); document.getElementById('sidebar').classList.remove('hidden'); };
    document.getElementById('close-sidebar').onclick = () => document.getElementById('sidebar').classList.add('hidden');
    document.getElementById('new-chat-btn').onclick = async () => { await fetch('/new_chat', {method:'POST'}); loadHistory(); document.getElementById('sidebar').classList.add('hidden'); };
    document.getElementById('expand-main-prompt').onclick = () => { modal.style.display='none'; openLargeEditor(null, 'prompt'); };
    document.getElementById('close-modal-x').onclick = () => modal.style.display = 'none';
    userInput.onkeydown = (e) => { if(e.key==='Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } };
});
// Last Updated: 2026-02-17 13:00:00