// State
let currentConversationId = null;
let activeCourseId = null;
let isStreaming = false;

// DOM elements
const courseList = document.getElementById('course-list');
const conversationList = document.getElementById('conversation-list');
const messages = document.getElementById('messages');
const messageInput = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const courseInfo = document.getElementById('course-info');
const loadBundleBtn = document.getElementById('load-bundle-btn');
const newConversationBtn = document.getElementById('new-conversation-btn');
const settingsBtn = document.getElementById('settings-btn');
const settingsModal = document.getElementById('settings-modal');
const closeSettings = document.getElementById('close-settings');
const saveSettings = document.getElementById('save-settings');
const bundleModal = document.getElementById('bundle-modal');
const closeBundleModal = document.getElementById('close-bundle-modal');
const submitBundlePath = document.getElementById('submit-bundle-path');
const bundlePathInput = document.getElementById('bundle-path-input');
const bundleFingerprintInput = document.getElementById('bundle-fingerprint-input');

// API helpers
async function api(method, path, body) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body) opts.body = JSON.stringify(body);
    const res = await fetch(path, opts);
    if (!res.ok) {
        const err = await res.json().catch(() => ({ error: res.statusText }));
        throw new Error(err.error || 'Request failed');
    }
    return res.json();
}

// Initialize
async function init() {
    await refreshCourses();
    await refreshConversations();
    showEmptyState();
}

function showEmptyState() {
    if (messages.children.length === 0) {
        messages.innerHTML = `
            <div class="empty-state">
                <h2>Welcome to Sheplet</h2>
                <p>Load a course file from your instructor to get started, then ask questions about your course materials.</p>
            </div>
        `;
    }
}

// Courses
async function refreshCourses() {
    try {
        const courses = await api('GET', '/api/courses');
        courseList.innerHTML = '';
        courses.forEach(c => {
            const li = document.createElement('li');
            li.textContent = c.course_name;
            li.title = `${c.model_name} v${c.version}`;
            if (c.is_active) {
                li.classList.add('active');
                activeCourseId = c.id;
                courseInfo.textContent = `${c.course_name} v${c.version}`;
            }
            li.addEventListener('click', () => switchCourse(c.id));
            courseList.appendChild(li);
        });
        if (courses.length === 0) {
            courseInfo.textContent = 'No course loaded';
        }
    } catch (e) {
        console.error('Failed to load courses:', e);
    }
}

async function switchCourse(courseId) {
    if (courseId === activeCourseId) return;
    courseInfo.innerHTML = '<span class="spinner"></span> Loading model...';
    try {
        await api('POST', '/api/courses/switch', { course_id: courseId });
        await refreshCourses();
        await refreshConversations();
        currentConversationId = null;
        messages.innerHTML = '';
        showEmptyState();
    } catch (e) {
        alert('Failed to switch course: ' + e.message);
        await refreshCourses();
    }
}

// Bundle loading
loadBundleBtn.addEventListener('click', () => {
    bundleModal.style.display = 'flex';
    bundlePathInput.value = '';
    bundleFingerprintInput.value = '';
    bundlePathInput.focus();
});

closeBundleModal.addEventListener('click', () => {
    bundleModal.style.display = 'none';
});

submitBundlePath.addEventListener('click', async () => {
    const path = bundlePathInput.value.trim();
    if (!path) return;
    const fingerprint = bundleFingerprintInput.value.trim();
    if (!fingerprint) {
        alert('Please enter the verification code from your instructor');
        bundleFingerprintInput.focus();
        return;
    }
    bundleModal.style.display = 'none';
    courseInfo.innerHTML = '<span class="spinner"></span> Loading bundle...';
    try {
        await api('POST', '/api/bundles/load', { path, trusted_fingerprint: fingerprint });
        await refreshCourses();
        await refreshConversations();
    } catch (e) {
        alert('Failed to load bundle: ' + e.message);
        await refreshCourses();
    }
});

bundlePathInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') submitBundlePath.click();
});

// Conversations
async function refreshConversations() {
    try {
        const query = activeCourseId ? `?course_id=${activeCourseId}` : '';
        const convs = await api('GET', `/api/conversations${query}`);
        conversationList.innerHTML = '';
        convs.forEach(c => {
            const li = document.createElement('li');
            if (c.id === currentConversationId) li.classList.add('active');

            const nameSpan = document.createElement('span');
            nameSpan.textContent = c.title + ` (${c.message_count})`;
            li.appendChild(nameSpan);

            const actions = document.createElement('span');

            const exportBtn = document.createElement('button');
            exportBtn.className = 'delete-btn';
            exportBtn.textContent = '\u2913';
            exportBtn.title = 'Export';
            exportBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                window.open(`/api/conversations/${c.id}/export`, '_blank');
            });
            actions.appendChild(exportBtn);

            const delBtn = document.createElement('button');
            delBtn.className = 'delete-btn';
            delBtn.textContent = '\u00d7';
            delBtn.title = 'Delete';
            delBtn.addEventListener('click', async (e) => {
                e.stopPropagation();
                await api('DELETE', `/api/conversations/${c.id}`);
                if (currentConversationId === c.id) {
                    currentConversationId = null;
                    messages.innerHTML = '';
                    showEmptyState();
                }
                await refreshConversations();
            });
            actions.appendChild(delBtn);

            li.appendChild(actions);
            li.addEventListener('click', () => loadConversation(c.id));
            conversationList.appendChild(li);
        });
    } catch (e) {
        console.error('Failed to load conversations:', e);
    }
}

async function loadConversation(id) {
    try {
        const conv = await api('GET', `/api/conversations/${id}`);
        currentConversationId = id;
        messages.innerHTML = '';
        conv.messages.forEach(m => {
            appendMessage(m.role === 'User' ? 'user' : 'assistant', m.content, m.citations, false);
        });
        await refreshConversations();
        scrollToBottom();
    } catch (e) {
        console.error('Failed to load conversation:', e);
    }
}

newConversationBtn.addEventListener('click', async () => {
    if (!activeCourseId) {
        alert('Please load a course first.');
        return;
    }
    try {
        const conv = await api('POST', '/api/conversations', { course_id: activeCourseId });
        currentConversationId = conv.id;
        messages.innerHTML = '';
        showEmptyState();
        await refreshConversations();
    } catch (e) {
        alert('Failed to create conversation: ' + e.message);
    }
});

// Chat
function appendMessage(role, content, citations, blocked) {
    // Remove empty state if present
    const empty = messages.querySelector('.empty-state');
    if (empty) empty.remove();

    const div = document.createElement('div');
    div.className = 'message';

    const roleDiv = document.createElement('div');
    roleDiv.className = `message-role ${role}`;
    roleDiv.textContent = role === 'user' ? 'You' : 'Tutor';
    div.appendChild(roleDiv);

    const contentDiv = document.createElement('div');
    if (blocked) {
        contentDiv.className = 'message-blocked';
    } else {
        contentDiv.className = 'message-content';
    }
    contentDiv.textContent = content;
    div.appendChild(contentDiv);

    if (citations && citations.length > 0) {
        const details = document.createElement('details');
        details.className = 'message-citations';
        const summary = document.createElement('summary');
        summary.textContent = `${citations.length} source${citations.length === 1 ? '' : 's'} used`;
        details.appendChild(summary);
        citations.forEach(c => {
            const citeDiv = document.createElement('div');
            citeDiv.className = 'citation';
            const header = document.createElement('div');
            header.className = 'citation-header';
            header.textContent = `${c.source_file} (passage ${c.chunk_index + 1})`;
            citeDiv.appendChild(header);
            const body = document.createElement('div');
            body.className = 'citation-body';
            body.textContent = c.text_snippet || '';
            citeDiv.appendChild(body);
            details.appendChild(citeDiv);
        });
        div.appendChild(details);
    }

    messages.appendChild(div);
    return contentDiv;
}

function scrollToBottom() {
    const chatArea = document.getElementById('chat-area');
    chatArea.scrollTop = chatArea.scrollHeight;
}

async function sendMessage() {
    const text = messageInput.value.trim();
    if (!text || isStreaming) return;
    if (!activeCourseId) {
        alert('Please load a course first.');
        return;
    }

    isStreaming = true;
    sendBtn.disabled = true;
    messageInput.value = '';

    appendMessage('user', text, null, false);
    scrollToBottom();

    // Create assistant message placeholder with pipeline status indicator
    const assistantContent = appendMessage('assistant', '', null, false);
    assistantContent.innerHTML = '';
    const statusBar = document.createElement('div');
    statusBar.className = 'pipeline-status';
    const stages = [
        { key: 'embedding', label: 'Understanding question' },
        { key: 'searching', label: 'Finding relevant materials' },
        { key: 'generating', label: 'Writing answer' },
    ];
    stages.forEach(stage => {
        const span = document.createElement('span');
        span.className = 'pipeline-stage';
        span.dataset.stage = stage.key;
        span.innerHTML = `<span class="stage-icon">&#9675;</span> ${stage.label}`;
        statusBar.appendChild(span);
    });
    assistantContent.appendChild(statusBar);
    let fullResponse = '';
    let citations = [];

    try {
        const body = { message: text };
        if (currentConversationId) body.conversation_id = currentConversationId;

        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ error: 'Request failed' }));
            assistantContent.textContent = 'Error: ' + (err.error || 'Unknown error');
            assistantContent.className = 'message-blocked';
            isStreaming = false;
            sendBtn.disabled = false;
            return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let currentEventType = null;

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // keep incomplete line

            for (const line of lines) {
                if (line.startsWith('event: ')) {
                    currentEventType = line.substring(7).trim();
                    continue;
                }
                if (line.startsWith('data: ')) {
                    const data = line.substring(6);

                    if (currentEventType === 'status') {
                        const pipelineStatus = assistantContent.querySelector('.pipeline-status');
                        if (pipelineStatus) {
                            pipelineStatus.querySelectorAll('.pipeline-stage').forEach(el => {
                                const stage = el.dataset.stage;
                                if (stage === data) {
                                    el.classList.add('active');
                                    el.classList.remove('completed');
                                    el.querySelector('.stage-icon').innerHTML = '<span class="spinner"></span>';
                                } else if (el.classList.contains('active')) {
                                    el.classList.remove('active');
                                    el.classList.add('completed');
                                    el.querySelector('.stage-icon').innerHTML = '&#10003;';
                                }
                            });
                        }
                        currentEventType = null;
                        continue;
                    }

                    if (currentEventType === 'done') {
                        try {
                            const parsed = JSON.parse(data);
                            if (parsed.conversation_id) {
                                currentConversationId = parsed.conversation_id;
                            }
                            if (parsed.blocked) {
                                assistantContent.className = 'message-blocked';
                            }
                        } catch (_) {}
                        // Mark "Generating" stage as completed
                        const genStage = assistantContent.querySelector('.pipeline-stage[data-stage="generating"]');
                        if (genStage && genStage.classList.contains('active')) {
                            genStage.classList.remove('active');
                            genStage.classList.add('completed');
                            genStage.querySelector('.stage-icon').innerHTML = '&#10003;';
                        }
                        currentEventType = null;
                        continue;
                    }

                    if (currentEventType === 'citations') {
                        try {
                            citations = JSON.parse(data);
                            const msgDiv = assistantContent.parentElement;
                            const details = document.createElement('details');
                            details.className = 'message-citations';
                            const summary = document.createElement('summary');
                            summary.textContent = `${citations.length} source${citations.length === 1 ? '' : 's'} used`;
                            details.appendChild(summary);
                            citations.forEach(c => {
                                const citeDiv = document.createElement('div');
                                citeDiv.className = 'citation';
                                const header = document.createElement('div');
                                header.className = 'citation-header';
                                header.textContent = `${c.source_file} (passage ${c.chunk_index + 1})`;
                                citeDiv.appendChild(header);
                                const body = document.createElement('div');
                                body.className = 'citation-body';
                                body.textContent = c.text_snippet || '';
                                citeDiv.appendChild(body);
                                details.appendChild(citeDiv);
                            });
                            msgDiv.appendChild(details);
                        } catch (_) {}
                        currentEventType = null;
                        continue;
                    }

                    if (currentEventType === 'error') {
                        assistantContent.textContent = 'Error: ' + data;
                        assistantContent.className = 'message-blocked';
                        currentEventType = null;
                        continue;
                    }

                    // Default: text token
                    fullResponse += data;
                    let responseDiv = assistantContent.querySelector('.response-text');
                    if (!responseDiv) {
                        responseDiv = document.createElement('div');
                        responseDiv.className = 'response-text';
                        assistantContent.appendChild(responseDiv);
                    }
                    responseDiv.textContent = fullResponse;
                    scrollToBottom();
                    currentEventType = null;
                }

                // Empty line resets event type (SSE spec)
                if (line === '') {
                    currentEventType = null;
                }
            }
        }
    } catch (e) {
        if (!fullResponse) {
            assistantContent.textContent = 'Error: ' + e.message;
            assistantContent.className = 'message-blocked';
        }
    }

    isStreaming = false;
    sendBtn.disabled = false;
    messageInput.focus();
    await refreshConversations();
}

sendBtn.addEventListener('click', sendMessage);
messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Auto-resize textarea
messageInput.addEventListener('input', () => {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
});

// Settings
settingsBtn.addEventListener('click', async () => {
    try {
        const settings = await api('GET', '/api/settings');
        document.getElementById('setting-strategy').value = settings.retrieval_strategy;
        document.getElementById('setting-k').value = settings.top_k;
        const thresholdInput = document.getElementById('setting-threshold');
        thresholdInput.value = settings.relevance_threshold;
        // The instructor sets a floor for relevance; it can be raised, not lowered
        thresholdInput.min = settings.min_relevance_threshold;
        settingsModal.style.display = 'flex';
    } catch (e) {
        alert('Please load a course first to view settings.');
    }
});

closeSettings.addEventListener('click', () => {
    settingsModal.style.display = 'none';
});

saveSettings.addEventListener('click', async () => {
    try {
        await api('PUT', '/api/settings', {
            retrieval_strategy: document.getElementById('setting-strategy').value,
            top_k: parseInt(document.getElementById('setting-k').value),
            relevance_threshold: parseFloat(document.getElementById('setting-threshold').value),
            mmr_lambda: parseFloat(document.getElementById('setting-lambda').value),
        });
        settingsModal.style.display = 'none';
    } catch (e) {
        alert('Failed to save settings: ' + e.message);
    }
});

// Close modals on backdrop click
[settingsModal, bundleModal].forEach(modal => {
    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.style.display = 'none';
    });
});

// Init
init();
