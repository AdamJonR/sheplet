// State
let activeProjectName = null;
let currentView = 'dashboard';

// DOM elements
const projectSelect = document.getElementById('project-select');
const headerInfo = document.getElementById('header-info');
const navList = document.getElementById('nav-list');
const taskIndicator = document.getElementById('task-indicator');
const taskIndicatorText = document.getElementById('task-indicator-text');

// API helper
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
    await refreshProjects();
    setupNavigation();
    setupNewProject();
    setupIngest();
    setupModel();
    setupFinetune();
    setupConfig();
    setupBundle();
}

// ---- Navigation ----
function setupNavigation() {
    navList.querySelectorAll('li').forEach(li => {
        li.addEventListener('click', () => switchView(li.dataset.view));
    });
}

function switchView(view) {
    currentView = view;
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.getElementById('view-' + view).classList.add('active');
    navList.querySelectorAll('li').forEach(li => {
        li.classList.toggle('active', li.dataset.view === view);
    });

    // Load view-specific data
    if (view === 'dashboard') refreshDashboard();
    if (view === 'config') loadConfig();
    if (view === 'finetune') refreshFinetuneFiles();
    if (view === 'bundle') refreshBundleInfo();
}

// ---- Projects ----
async function refreshProjects() {
    try {
        const projects = await api('GET', '/api/projects');
        projectSelect.innerHTML = '<option value="">Select a project...</option>';
        projects.forEach(p => {
            const opt = document.createElement('option');
            opt.value = p.name;
            opt.textContent = `${p.course_name} (${p.name})`;
            if (p.is_active) {
                opt.selected = true;
                activeProjectName = p.name;
                headerInfo.textContent = `${p.course_name} v${p.version}`;
            }
            projectSelect.appendChild(opt);
        });
        if (!activeProjectName) {
            headerInfo.textContent = 'No project selected';
        }
        refreshDashboard();
    } catch (e) {
        console.error('Failed to load projects:', e);
    }
}

projectSelect.addEventListener('change', async () => {
    const name = projectSelect.value;
    if (!name) return;
    try {
        await api('POST', '/api/projects/select', { name });
        activeProjectName = name;
        await refreshProjects();
    } catch (e) {
        alert('Failed to select project: ' + e.message);
    }
});

function setupNewProject() {
    const modal = document.getElementById('new-project-modal');
    const closeBtn = document.getElementById('close-new-project');
    const createBtn = document.getElementById('create-project-btn');
    const newBtn = document.getElementById('new-project-btn');

    newBtn.addEventListener('click', () => {
        modal.style.display = 'flex';
        document.getElementById('np-course-name').value = '';
        document.getElementById('np-dir-name').value = '';
        document.getElementById('np-course-name').focus();
    });

    closeBtn.addEventListener('click', () => modal.style.display = 'none');
    modal.addEventListener('click', (e) => { if (e.target === modal) modal.style.display = 'none'; });

    // Auto-generate directory name from course name
    document.getElementById('np-course-name').addEventListener('input', (e) => {
        const dirName = e.target.value
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, '-')
            .replace(/^-|-$/g, '');
        document.getElementById('np-dir-name').value = dirName;
    });

    createBtn.addEventListener('click', async () => {
        const courseName = document.getElementById('np-course-name').value.trim();
        const dirName = document.getElementById('np-dir-name').value.trim();
        if (!courseName || !dirName) {
            alert('Please fill in both fields');
            return;
        }
        try {
            await api('POST', '/api/projects', { course_name: courseName, directory_name: dirName });
            modal.style.display = 'none';
            await refreshProjects();
        } catch (e) {
            alert('Failed to create project: ' + e.message);
        }
    });
}

// ---- Dashboard ----
async function refreshDashboard() {
    const noProject = document.getElementById('no-project-state');
    const content = document.getElementById('dashboard-content');
    const checklist = document.getElementById('status-checklist');

    if (!activeProjectName) {
        noProject.style.display = 'block';
        content.style.display = 'none';
        return;
    }

    try {
        const data = await api('GET', '/api/projects/active');
        if (!data.project) {
            noProject.style.display = 'block';
            content.style.display = 'none';
            return;
        }
        noProject.style.display = 'none';
        content.style.display = 'block';

        const s = data.project.status;
        checklist.innerHTML = '';

        const items = [
            { done: true, label: 'Project initialized', detail: data.project.course_name },
            { done: s.has_config, label: 'Configuration set', detail: s.has_config ? 'config.json present' : 'Using defaults' },
            { done: s.has_database, label: 'Documents ingested', detail: s.has_database ? 'Vector database populated' : 'No documents ingested yet' },
            { done: s.has_model, label: 'Model downloaded', detail: s.model_name ? s.model_name : 'No model downloaded' },
            { done: s.has_finetune_data, label: 'Fine-tuning data prepared', detail: s.finetune_files.length ? s.finetune_files.join(', ') : 'No JSONL files' },
            { done: s.has_adapter, label: 'Fine-tuning complete', detail: s.has_adapter ? 'adapter.safetensors present' : 'Not fine-tuned yet' },
            { done: !!s.build_timestamp, label: 'Bundle created', detail: s.build_timestamp ? `Build: ${s.build_timestamp}` : 'No bundle created yet' },
        ];

        items.forEach(item => {
            const div = document.createElement('div');
            div.className = `checklist-item ${item.done ? 'done' : 'pending'}`;
            div.innerHTML = `
                <span class="check-icon">${item.done ? '\u2713' : '\u25CB'}</span>
                <span class="check-label">${item.label}</span>
                <span class="check-detail">${item.detail}</span>
            `;
            checklist.appendChild(div);
        });
    } catch (e) {
        console.error('Failed to load dashboard:', e);
    }
}

// ---- SSE Task Subscription ----
function subscribeToTask(taskId, progressEl, onDone) {
    progressEl.style.display = 'block';
    progressEl.innerHTML = '';

    const stepsDiv = document.createElement('div');
    stepsDiv.className = 'progress-steps';
    const logDiv = document.createElement('div');
    logDiv.className = 'progress-log';
    logDiv.style.display = 'none';
    progressEl.appendChild(stepsDiv);
    progressEl.appendChild(logDiv);

    const steps = {};

    const es = new EventSource(`/api/tasks/${taskId}/stream`);

    es.addEventListener('step', (e) => {
        const data = JSON.parse(e.data);
        const stepName = data.step;
        // Mark any previously active step as stale
        Object.values(steps).forEach(el => {
            if (el.classList.contains('active')) {
                el.classList.remove('active');
                el.classList.add('completed');
                el.querySelector('.step-icon').innerHTML = '\u2713';
            }
        });

        const stepDiv = document.createElement('div');
        stepDiv.className = 'progress-step active';
        stepDiv.innerHTML = `<span class="step-icon"><span class="spinner"></span></span><span>${stepName}</span>`;
        stepsDiv.appendChild(stepDiv);
        steps[stepName] = stepDiv;
    });

    es.addEventListener('step_done', (e) => {
        const data = JSON.parse(e.data);
        const el = steps[data.step];
        if (el) {
            el.classList.remove('active');
            el.classList.add('completed');
            el.querySelector('.step-icon').innerHTML = '\u2713';
            if (data.detail) {
                const detailSpan = document.createElement('span');
                detailSpan.className = 'step-detail';
                detailSpan.textContent = ' \u2014 ' + data.detail;
                el.appendChild(detailSpan);
            }
        }
    });

    es.addEventListener('progress', (e) => {
        const data = JSON.parse(e.data);
        const el = steps[data.step];
        if (el) {
            let pct = Math.round((data.current / data.total) * 100);
            el.querySelector('span:last-child').textContent = `${data.step} (${pct}%)`;
        }
    });

    es.addEventListener('log', (e) => {
        logDiv.style.display = 'block';
        const line = document.createElement('div');
        line.className = 'log-line';
        line.textContent = e.data;
        logDiv.appendChild(line);
        logDiv.scrollTop = logDiv.scrollHeight;
    });

    es.addEventListener('done', (e) => {
        es.close();
        const data = JSON.parse(e.data);
        const resultDiv = document.createElement('div');
        if (data.success) {
            resultDiv.className = 'progress-result success';
            resultDiv.textContent = 'Completed successfully';
        } else {
            resultDiv.className = 'progress-result failure';
            resultDiv.textContent = 'Failed: ' + (data.error || 'Unknown error');
        }
        progressEl.appendChild(resultDiv);
        hideTaskIndicator();
        if (onDone) onDone(data.success);
    });

    es.onerror = () => {
        es.close();
        hideTaskIndicator();
    };

    showTaskIndicator();
    return es;
}

function showTaskIndicator() {
    taskIndicator.style.display = 'flex';
}

function hideTaskIndicator() {
    taskIndicator.style.display = 'none';
}

// ---- Ingest ----
function setupIngest() {
    const btn = document.getElementById('ingest-btn');
    btn.addEventListener('click', async () => {
        const sources = document.getElementById('ingest-sources').value.trim();
        if (!sources) { alert('Enter the source documents directory path'); return; }
        if (!activeProjectName) { alert('Select a project first'); return; }

        btn.disabled = true;
        try {
            const { task_id } = await api('POST', '/api/ingest', { sources_path: sources });
            subscribeToTask(task_id, document.getElementById('ingest-progress'), () => {
                btn.disabled = false;
                refreshDashboard();
            });
        } catch (e) {
            alert('Failed to start ingestion: ' + e.message);
            btn.disabled = false;
        }
    });
}

// ---- Model ----
function setupModel() {
    const btn = document.getElementById('model-btn');
    btn.addEventListener('click', async () => {
        if (!activeProjectName) { alert('Select a project first'); return; }
        const name = document.getElementById('model-name').value;

        btn.disabled = true;
        try {
            const { task_id } = await api('POST', '/api/model/download', { name });
            subscribeToTask(task_id, document.getElementById('model-progress'), () => {
                btn.disabled = false;
                refreshDashboard();
            });
        } catch (e) {
            alert('Failed to start model download: ' + e.message);
            btn.disabled = false;
        }
    });
}

// ---- Fine-tune ----
function setupFinetune() {
    const genBtn = document.getElementById('gen-templates-btn');
    const refreshBtn = document.getElementById('refresh-files-btn');
    const ftBtn = document.getElementById('finetune-btn');

    genBtn.addEventListener('click', async () => {
        if (!activeProjectName) { alert('Select a project first'); return; }
        try {
            const result = await api('POST', '/api/templates/generate');
            document.getElementById('gen-templates-status').textContent = result.message;
            await refreshFinetuneFiles();
        } catch (e) {
            alert('Failed to generate templates: ' + e.message);
        }
    });

    refreshBtn.addEventListener('click', refreshFinetuneFiles);

    ftBtn.addEventListener('click', async () => {
        if (!activeProjectName) { alert('Select a project first'); return; }
        const method = document.getElementById('ft-method').value;
        const dataFile = document.getElementById('ft-data').value;
        if (!dataFile) { alert('Select a data file'); return; }

        const body = { method, data_file: dataFile };
        const lr = document.getElementById('ft-lr').value;
        const epochs = document.getElementById('ft-epochs').value;
        if (lr) body.learning_rate = parseFloat(lr);
        if (epochs) body.epochs = parseInt(epochs);

        ftBtn.disabled = true;
        try {
            const { task_id } = await api('POST', '/api/finetune', body);
            subscribeToTask(task_id, document.getElementById('finetune-progress'), () => {
                ftBtn.disabled = false;
                refreshDashboard();
            });
        } catch (e) {
            alert('Failed to start fine-tuning: ' + e.message);
            ftBtn.disabled = false;
        }
    });
}

async function refreshFinetuneFiles() {
    if (!activeProjectName) return;
    try {
        const files = await api('GET', '/api/templates/files');
        const select = document.getElementById('ft-data');
        select.innerHTML = '';
        if (files.length === 0) {
            select.innerHTML = '<option value="">No JSONL files found</option>';
        } else {
            files.forEach(f => {
                const opt = document.createElement('option');
                opt.value = f;
                opt.textContent = f;
                select.appendChild(opt);
            });
        }
    } catch (e) {
        console.error('Failed to refresh files:', e);
    }
}

// ---- Config ----
function setupConfig() {
    const thresholdInput = document.getElementById('cfg-threshold');
    const thresholdVal = document.getElementById('cfg-threshold-val');
    const lambdaInput = document.getElementById('cfg-lambda');
    const lambdaVal = document.getElementById('cfg-lambda-val');

    thresholdInput.addEventListener('input', () => thresholdVal.textContent = thresholdInput.value);
    lambdaInput.addEventListener('input', () => lambdaVal.textContent = lambdaInput.value);

    document.getElementById('save-config-btn').addEventListener('click', async () => {
        if (!activeProjectName) { alert('Select a project first'); return; }
        try {
            await api('PUT', '/api/config', {
                system_prompt: document.getElementById('cfg-prompt').value,
                retrieval_strategy: document.getElementById('cfg-strategy').value,
                top_k: parseInt(document.getElementById('cfg-topk').value),
                relevance_threshold: parseFloat(thresholdInput.value),
                mmr_lambda: parseFloat(lambdaInput.value),
            });
            document.getElementById('config-status').textContent = 'Saved!';
            setTimeout(() => document.getElementById('config-status').textContent = '', 2000);
            refreshDashboard();
        } catch (e) {
            alert('Failed to save config: ' + e.message);
        }
    });
}

async function loadConfig() {
    if (!activeProjectName) return;
    try {
        const config = await api('GET', '/api/config');
        document.getElementById('cfg-prompt').value = config.system_prompt;
        document.getElementById('cfg-strategy').value = config.retrieval_strategy;
        document.getElementById('cfg-topk').value = config.top_k;
        document.getElementById('cfg-threshold').value = config.relevance_threshold;
        document.getElementById('cfg-threshold-val').textContent = config.relevance_threshold;
        document.getElementById('cfg-lambda').value = config.mmr_lambda;
        document.getElementById('cfg-lambda-val').textContent = config.mmr_lambda;
    } catch (e) {
        console.error('Failed to load config:', e);
    }
}

// ---- Bundle ----
function setupBundle() {
    const btn = document.getElementById('bundle-btn');
    btn.addEventListener('click', async () => {
        if (!activeProjectName) { alert('Select a project first'); return; }
        const outputPath = document.getElementById('bundle-output').value.trim();
        if (!outputPath) { alert('Enter the output file path'); return; }

        const bump = document.getElementById('bundle-bump').checked;

        btn.disabled = true;
        try {
            const { task_id } = await api('POST', '/api/bundle', {
                output_path: outputPath,
                bump_version: bump,
            });
            subscribeToTask(task_id, document.getElementById('bundle-progress'), () => {
                btn.disabled = false;
                refreshDashboard();
                refreshBundleInfo();
            });
        } catch (e) {
            alert('Failed to start bundling: ' + e.message);
            btn.disabled = false;
        }
    });
}

async function refreshBundleInfo() {
    const infoBox = document.getElementById('bundle-info');
    if (!activeProjectName) {
        infoBox.innerHTML = '<span style="color:var(--text-muted)">Select a project to see bundle info.</span>';
        return;
    }
    try {
        const data = await api('GET', '/api/projects/active');
        if (!data.project) return;
        const s = data.project.status;
        infoBox.innerHTML = `
            <div class="info-row"><span class="info-label">Version:</span> ${data.project.version}</div>
            <div class="info-row"><span class="info-label">Model:</span> ${s.model_name || 'Not set'}</div>
            <div class="info-row"><span class="info-label">Adapter:</span> ${s.has_adapter ? 'Present' : 'Not trained'}</div>
            <div class="info-row"><span class="info-label">Database:</span> ${s.has_database ? 'Populated' : 'Empty'}</div>
            <div class="info-row"><span class="info-label">Last build:</span> ${s.build_timestamp || 'Never'}</div>
        `;
    } catch (e) {
        console.error('Failed to load bundle info:', e);
    }
}

// ---- Start ----
init();
