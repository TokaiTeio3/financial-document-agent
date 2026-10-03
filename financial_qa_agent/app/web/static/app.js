const form = document.querySelector('#questionForm');
const input = document.querySelector('#questionInput');
const submitButton = document.querySelector('#submitButton');
const charCount = document.querySelector('#charCount');
const runPanel = document.querySelector('#runPanel');
const answerPanel = document.querySelector('#answerPanel');
const errorPanel = document.querySelector('#errorPanel');
const traceList = document.querySelector('#traceList');
const runStatus = document.querySelector('#runStatus');
const tokenModels = document.querySelector('#tokenModels');
const tokenGrandTotal = document.querySelector('#tokenGrandTotal');

function textElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  element.textContent = text;
  return element;
}

function resetRun() {
  traceList.replaceChildren();
  answerPanel.hidden = true;
  errorPanel.hidden = true;
  runPanel.hidden = false;
  runStatus.textContent = '执行中';
  runStatus.classList.remove('done');
  tokenGrandTotal.textContent = '0';
  tokenModels.replaceChildren(textElement('span', 'token-empty', '等待模型调用'));
}

function renderTokenUsage(data) {
  tokenGrandTotal.textContent = Number(data.grand_total || 0).toLocaleString('zh-CN');
  const cards = Object.entries(data.models || {}).map(([model, usage]) => {
    const card = document.createElement('article');
    card.className = 'token-card';
    card.append(textElement('strong', '', model));
    const metrics = document.createElement('div');
    metrics.className = 'token-metrics';
    metrics.append(textElement('span', '', `输入 ${Number(usage.input_tokens || 0).toLocaleString('zh-CN')}`));
    metrics.append(textElement('span', '', `输出 ${Number(usage.output_tokens || 0).toLocaleString('zh-CN')}`));
    metrics.append(textElement('span', '', `${usage.calls || 0} 次调用`));
    card.append(metrics);
    return card;
  });
  tokenModels.replaceChildren(...cards);
}

function addTrace(event) {
  const item = document.createElement('article');
  item.className = 'trace-item';
  item.append(textElement('span', 'trace-index', String(event.step).padStart(2, '0')));
  item.append(textElement('h3', '', event.title));
  item.append(textElement('p', '', event.summary));
  if (event.details && Object.keys(event.details).length) {
    const details = document.createElement('details');
    details.append(textElement('summary', '', '查看输入、输出与公开理由'));
    details.append(textElement('pre', '', JSON.stringify(event.details, null, 2)));
    item.append(details);
  }
  traceList.append(item);
  item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function populateList(sectionId, listId, values) {
  const section = document.querySelector(sectionId);
  const list = document.querySelector(listId);
  list.replaceChildren(...values.map(value => textElement('li', '', value)));
  section.hidden = values.length === 0;
}

function showAnswer(answer) {
  document.querySelector('#answerText').textContent = answer.answer;
  document.querySelector('#confidenceBadge').textContent = `CONFIDENCE / ${answer.confidence.toUpperCase()}`;
  populateList('#keyPointsSection', '#keyPoints', answer.key_points || []);
  populateList('#limitationsSection', '#limitations', answer.limitations || []);

  const citationSection = document.querySelector('#citationsSection');
  const citations = document.querySelector('#citations');
  citations.replaceChildren(...(answer.citations || []).map(citation => {
    const card = document.createElement('article');
    card.className = 'citation';
    const location = citation.page ? ` · P.${citation.page}` : '';
    card.append(textElement('strong', '', `${citation.document_id}${location} · ${citation.source}`));
    card.append(textElement('p', '', citation.quote));
    return card;
  }));
  citationSection.hidden = !(answer.citations || []).length;
  answerPanel.hidden = false;
}

function showError(message) {
  errorPanel.textContent = `运行失败：${message}`;
  errorPanel.hidden = false;
  runStatus.textContent = '失败';
}

async function runAgent(question) {
  resetRun();
  submitButton.disabled = true;
  try {
    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === 'trace') addTrace(event.data);
        if (event.type === 'token_usage') renderTokenUsage(event.data);
        if (event.type === 'answer') showAnswer(event.data);
        if (event.type === 'error') throw new Error(event.data.message);
        if (event.type === 'done') {
          runStatus.textContent = '已完成';
          runStatus.classList.add('done');
        }
      }
      if (done) break;
    }
  } catch (error) {
    showError(error instanceof Error ? error.message : String(error));
  } finally {
    submitButton.disabled = false;
  }
}

form.addEventListener('submit', event => {
  event.preventDefault();
  const question = input.value.trim();
  if (question) runAgent(question);
});
input.addEventListener('input', () => { charCount.textContent = `${input.value.length} / 4000`; });
input.addEventListener('keydown', event => {
  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) form.requestSubmit();
});
document.querySelectorAll('[data-question]').forEach(button => button.addEventListener('click', () => {
  input.value = button.dataset.question;
  input.dispatchEvent(new Event('input'));
  input.focus();
}));

fetch('/api/meta').then(response => response.json()).then(meta => {
  const label = meta.mode === 'demo' ? 'DEMO MODE · 本地检索' : `${meta.planner_model} → ${meta.tool_model} · 已连接`;
  document.querySelector('#modeLabel').textContent = label;
  document.querySelector('.pulse').classList.add('online');
}).catch(() => { document.querySelector('#modeLabel').textContent = '连接失败'; });
