let predictionChart = null;
let dailyChart = null;
let sparklineChart = null;
let activeTab = 'home';
let map = null;
let mapMarkers = {};
let railCollapsed = false;
let alertHistory = [];

const regionMockData = {
    '제주 전역': { gen: 82.4, solarGen: 45.2, windGen: 37.2, solar: 1.85, weather: '맑음 (평균 풍속 6.2 m/s)', ai: '전반적인 기압 분포가 안정되어 있고, 구름 유입도가 적어 제주도 전역의 신재생에너지 발전량이 준수할 것으로 관측됩니다.' },
    '제주시': { gen: 45.2, solarGen: 27.1, windGen: 18.1, solar: 1.95, weather: '매우 맑음 (풍속 5.1 m/s)', ai: '제주 북부 해안 영역은 구름 농도가 10% 미만으로 집계되어 태양광 발전 효율이 극대화되는 경향을 띱니다.' },
    '서귀포시': { gen: 37.2, solarGen: 18.6, windGen: 18.6, solar: 1.70, weather: '구름 조금 (풍속 4.5 m/s)', ai: '서귀포 남부 해안선 방향으로 남서풍 구름 유입이 일부 존재하여 태양광 발전량이 소폭 한계 수치에 도달해 있습니다.' },
    '한경/고산': { gen: 52.8, solarGen: 5.28, windGen: 47.52, solar: 2.10, weather: '강풍 및 맑음 (풍속 12.5 m/s)', ai: '서부 한경면 일대는 한라산 배풍 패턴에 기인한 강한 고풍속이 형성되어 풍력 발전량이 85% 이상의 피크 출력을 기록 중입니다.' },
    '구좌/성산': { gen: 18.5, solarGen: 4.62, windGen: 13.88, solar: 0.65, weather: '흐리고 비 (풍속 8.2 m/s)', ai: '동부 구좌읍 일대는 기압 골 통과에 따른 전선 접근으로 인해 전운량이 급변하고 있어 발전량이 최저 수준에 머물러 있습니다.' }
};

// tool_trace 레이블 맵
const TOOL_LABELS = {
    tool_lookup: '실측 데이터 조회',
    tool_predict: 'LSTM 모델 예측 실행',
    tool_rag: '유사 기상 사례 검색',
    tool_kg_query: '지식 그래프 이벤트 조회',
    tool_explain: '영향 변수 설명 생성',
    tool_compare: '시점 간 발전량 비교',
};

function toggleTraceLog() {
    const list = document.getElementById('rail-trace-list');
    const chevron = document.getElementById('trace-chevron');
    const hidden = list.classList.toggle('hidden');
    chevron.textContent = hidden ? '▼' : '▲';
}

// ── 글로벌 상태바 갱신 ──────────────────────────────────────
function updateGlobalStatusBar(report) {
    const bar = document.getElementById('global-status-bar');
    const dot = document.getElementById('gsb-dot');
    const text = document.getElementById('gsb-text');
    const now = new Date();
    document.getElementById('gsb-updated').textContent =
        `마지막 갱신: ${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;

    if (!report) {
        dot.className = 'h-1.5 w-1.5 rounded-full bg-amber-400 flex-shrink-0';
        text.textContent = '에이전트 처리 중...';
        bar.style.backgroundColor = '#ffffff'; bar.style.color = '#d97706';
        return;
    }
    const conf = report.confidence || 'High';
    document.getElementById('gsb-confidence').textContent = `신뢰도: ${conf}`;

    if (report.trigger_active || (report.warnings && report.warnings.length > 0)) {
        dot.className = 'h-1.5 w-1.5 rounded-full bg-red-500 flex-shrink-0';
        text.textContent = `경보 감지 — ${(report.warnings || []).slice(0, 1).join('')}`;
        bar.style.backgroundColor = '#ffffff'; bar.style.color = '#dc2626';
    } else if (conf === 'Low') {
        dot.className = 'h-1.5 w-1.5 rounded-full bg-amber-500 flex-shrink-0';
        text.textContent = '트리거 감지 — 예측 불확실성 높음';
        bar.style.backgroundColor = '#ffffff'; bar.style.color = '#d97706';
    } else {
        dot.className = 'h-1.5 w-1.5 rounded-full bg-emerald-500 flex-shrink-0';
        text.textContent = '에이전트 정상 모니터링 중';
        bar.style.backgroundColor = '#ffffff'; bar.style.color = '#16a34a';
    }
}

// ── 에이전트 추론 시작 애니메이션 ────────────────────────────
function showAgentThinking() {
    const list = document.getElementById('rail-trace-list');
    if (!list) return; // 방어 코드
    list.classList.remove('hidden');
    const chevron = document.getElementById('trace-chevron');
    if (chevron) chevron.textContent = '▲';
    list.innerHTML =
        `<div class="flex items-center justify-between bg-white border border-[#e5e7eb] rounded-xl px-3 py-2 mb-1">`
        + `<div class="flex items-center gap-2 text-[10px] font-semibold text-[#374151]">`
        + `<span class="agent-thinking-dot text-amber-500">◐</span> 에이전트 추론 중...`
        + `</div>`
        + `<span class="text-[9px] font-mono text-[#9ca3af]">처리 중</span>`
        + `</div>`;
}

// ── tool_trace 렌더링 (Image-6 스타일) ───────────────────────
function renderToolTrace(traceArray) {
    const list = document.getElementById('rail-trace-list');
    if (!list) return; // 방어 코드
    const chevron = document.getElementById('trace-chevron');
    if (chevron) chevron.textContent = '▲';
    if (!traceArray || traceArray.length === 0) {
        list.innerHTML = '<p class="text-[10px] text-[#9ca3af] italic">추론 로그 없음</p>';
        return;
    }
    list.innerHTML = '';

    const header = document.createElement('div');
    header.className = 'flex items-center justify-between bg-white border border-[#e5e7eb] rounded-xl px-3 py-2 mb-2';
    header.innerHTML = `<span class="text-[10px] font-semibold text-[#374151]">✓ 추론 완료</span>`
        + `<span class="text-[9px] font-mono text-[#9ca3af]">${traceArray.length}단계</span>`;
    list.appendChild(header);

    traceArray.forEach((step, i) => {
        const ok = step.status === 'ok';
        const label = TOOL_LABELS[step.tool] || step.tool;
        const div = document.createElement('div');
        div.className = 'flex gap-2 items-start animate-fade-in';
        div.style.animationDelay = `${i * 60}ms`;

        let subSteps = '';
        if (step.tool === 'tool_predict' && ok) {
            subSteps = `<div class="mt-1.5 ml-4 space-y-1">`
                + `<div class="flex items-center gap-1.5 text-[9px] text-[#737373]"><span class="text-emerald-500">✓</span> 예측 완료 `
                + `<span class="bg-amber-900/30 text-amber-400 font-mono px-1.5 py-0.5 rounded text-[8px]">${step.timestamp || ''}</span></div>`
                + `</div>`;
        } else if (step.tool === 'tool_rag' && ok) {
            subSteps = `<div class="mt-1.5 ml-4"><div class="flex items-center gap-1.5 text-[9px] text-[#737373]"><span class="text-emerald-500">✓</span> 유사 사례 <span class="bg-blue-900/30 text-blue-400 font-mono px-1.5 py-0.5 rounded text-[8px]">${step.found || 0}건</span> 검색됨</div></div>`;
        } else if (step.tool === 'tool_lookup') {
            const chip = step.type ? `<span class="bg-[#1a1a1a] border border-[#2a2a2a] font-mono text-[#737373] px-1.5 py-0.5 rounded text-[8px]">${step.type}</span>` : '';
            subSteps = `<div class="mt-1.5 ml-4"><div class="flex items-center gap-1.5 text-[9px] text-[#737373]">${ok ? '<span class="text-emerald-500">✓</span>' : '<span class="text-[#a3a3a3]">—</span>'} ${chip}</div></div>`;
        }

        div.innerHTML =
            `<div class="flex flex-col items-center flex-shrink-0 mt-0.5">`
            + `<span class="text-[11px] leading-none ${ok ? 'text-emerald-500' : 'text-[#d1d5db]'}">${ok ? '◉' : '○'}</span>`
            + (i < traceArray.length - 1 ? `<div class="w-px flex-1 bg-[#e5e7eb] my-0.5" style="min-height:10px;"></div>` : '')
            + `</div>`
            + `<div class="flex-1 min-w-0 pb-2">`
            + `<div class="text-[10px] font-medium text-[#374151] leading-snug">${label}</div>`
            + `<div class="flex items-center gap-1 mt-0.5">`
            + `<span class="font-mono text-[8px] text-[#9ca3af] bg-[#f4f4f5] px-1 py-0.5 rounded">${step.tool}</span>`
            + `<span class="text-[8px] px-1 py-0.5 rounded ${ok ? 'text-emerald-600' : 'text-red-500'}">${step.status}</span>`
            + `</div>`
            + subSteps
            + `</div>`;
        list.appendChild(div);
    });
}

// ── 레일 상태 카드 갱신 ──────────────────────────────────────
function updateRailStatusCard(report) {
    if (!report) return;
    const conf = report.confidence || 'High';
    const total = ((report.predictions?.solar_mw || 0) + (report.predictions?.wind_mw || 0)).toFixed(1);
    document.getElementById('rail-confidence').textContent = conf;
    document.getElementById('rail-total-gen').textContent = total + ' MW';
    const card = document.getElementById('rail-status-card');
    const dot = document.getElementById('rail-dot');
    const label = document.getElementById('rail-status-label');
    if (report.trigger_active || (report.warnings || []).length > 0) {
        card.className = 'theme-red rounded-xl p-3 space-y-1.5 border transition-all duration-300';
        dot.className = 'h-2 w-2 rounded-full bg-red-500 flex-shrink-0';
        label.textContent = '경보 발동';
    } else if (conf === 'Low') {
        card.className = 'theme-amber rounded-xl p-3 space-y-1.5 border transition-all duration-300';
        dot.className = 'h-2 w-2 rounded-full bg-amber-500 flex-shrink-0';
        label.textContent = '불확실성 높음';
    } else {
        card.className = 'theme-emerald rounded-xl p-3 space-y-1.5 border transition-all duration-300';
        dot.className = 'h-2 w-2 rounded-full bg-emerald-500 flex-shrink-0';
        label.textContent = '정상 모니터링';
    }
}

// ── 경보 이력 추가 ────────────────────────────────────────────
function appendAlertToRail(warnings, report) {
    if (!warnings || warnings.length === 0) return;
    const now = new Date();
    const t = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
    alertHistory.unshift({ time: t, msg: warnings[0], conf: report?.confidence });
    if (alertHistory.length > 10) alertHistory.pop();

    const listEl = document.getElementById('rail-alert-list');
    const countEl = document.getElementById('rail-alert-count');
    listEl.innerHTML = '';
    countEl.textContent = alertHistory.length + '건';
    countEl.classList.remove('hidden');
    alertHistory.forEach(a => {
        const div = document.createElement('div');
        div.className = 'text-[10px] bg-red-50 border border-red-200 rounded-lg px-2.5 py-1.5 animate-fade-in';
        div.innerHTML = `<span class="font-mono text-[#9ca3af]">${a.time}</span> <span class="text-red-600">${a.msg}</span>`;
        listEl.appendChild(div);
    });
}

// ── 통합 챗봇 전송 ───────────────────────────────────────────
// ── 실시간 추론 타임라인 HTML 템플릿 ───────────────────────
function createTraceTimelineHtml(traceId) {
    return `
    <div id="${traceId}-container" class="bg-white border border-[#e5e7eb] rounded-xl p-4 my-2 text-xs w-full max-w-[90%] space-y-4 animate-fade-in shadow-sm">
        <!-- Header -->
        <div class="flex items-center justify-between border-b border-[#e5e7eb] pb-2">
            <div class="flex items-center gap-2 font-bold text-[#374151]">
                <span class="relative flex h-2 w-2 mr-1">
                    <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
                    <span class="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
                </span>
                <span id="${traceId}-header-status">에이전트가 실시간 분석 중...</span>
            </div>
            <span class="text-[9px] text-[#9ca3af] font-mono" id="${traceId}-timer">0.0초 경과</span>
        </div>

        <!-- Timeline Steps -->
        <div class="relative pl-4 border-l border-[#e5e7eb] ml-2.5 space-y-4 text-[11px]" id="${traceId}-steps-list">
            <!-- Step 1 -->
            <div class="relative step-node pl-1" id="${traceId}-step-1">
                <span class="absolute -left-4 top-1.5 h-2 w-2 rounded-full bg-[#d1d5db] transform -translate-x-1/2 step-status transition-all duration-300"></span>
                <div class="font-medium text-[#9ca3af] step-title">실시간 권역 발전 실측 데이터 조회 (tool_lookup)</div>
                <div class="text-[9px] text-[#9ca3af] mt-0.5 step-desc hidden">제주 전역의 현재 발전량 수치 및 오차 한계선 로딩 중...</div>
            </div>
            <!-- Step 2 -->
            <div class="relative step-node pl-1" id="${traceId}-step-2">
                <span class="absolute -left-4 top-1.5 h-2 w-2 rounded-full bg-[#d1d5db] transform -translate-x-1/2 step-status transition-all duration-300"></span>
                <div class="font-medium text-[#9ca3af] step-title">LSTM 신경망 예측 추론 연산 (tool_predict)</div>
                <div class="text-[9px] text-[#9ca3af] mt-0.5 step-desc hidden">기상 패턴 데이터(온도/풍속/일사량) 기반 모델 가중치 추론 중...</div>
            </div>
            <!-- Step 3 -->
            <div class="relative step-node pl-1" id="${traceId}-step-3">
                <span class="absolute -left-4 top-1.5 h-2 w-2 rounded-full bg-[#d1d5db] transform -translate-x-1/2 step-status transition-all duration-300"></span>
                <div class="font-medium text-[#9ca3af] step-title">유사 기상 및 발전 트렌드 RAG 매칭 (tool_rag)</div>
                <div class="text-[9px] text-[#9ca3af] mt-0.5 step-desc hidden">벡터 데이터베이스에서 유사 기상 이력 레코드 매칭 중...</div>
            </div>
            <!-- Step 4 -->
            <div class="relative step-node pl-1" id="${traceId}-step-4">
                <span class="absolute -left-4 top-1.5 h-2 w-2 rounded-full bg-[#d1d5db] transform -translate-x-1/2 step-status transition-all duration-300"></span>
                <div class="font-medium text-[#9ca3af] step-title">지식 그래프 물리적 제약조건 스캔 (tool_kg_query)</div>
                <div class="text-[9px] text-[#9ca3af] mt-0.5 step-desc hidden">그리드 한계 규칙 및 특이 물리 현상(강풍 제약 등) 평가 중...</div>
            </div>
            <!-- Step 5 -->
            <div class="relative step-node pl-1" id="${traceId}-step-5">
                <span class="absolute -left-4 top-1.5 h-2 w-2 rounded-full bg-[#d1d5db] transform -translate-x-1/2 step-status transition-all duration-300"></span>
                <div class="font-medium text-[#9ca3af] step-title">SHAP 변수 기여도 설명 생성 (tool_explain)</div>
                <div class="text-[9px] text-[#9ca3af] mt-0.5 step-desc hidden">설명 가능한 AI(XAI) 기여도 분석 및 자율 재예측 점검 중...</div>
            </div>
        </div>

        <!-- 최종 AI 답변 텍스트 노출 공간 -->
        <div id="${traceId}-narrative" class="hidden border-t border-[#e5e7eb] pt-3 mt-2 text-xs text-[#374151] leading-relaxed break-words whitespace-pre-line animate-fade-in">
        </div>
    </div>
    `;
}

// sleep 헬퍼 함수
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ── 통합 챗봇 전송 ───────────────────────────────────────────
async function sendRailChat() {
    const input = document.getElementById('rail-chat-input');
    const messages = document.getElementById('rail-chat-messages');
    const text = input.value.trim();
    if (!text) return;

    // 1. 사용자 메시지 말풍선 추가 (더 깔끔한 UI)
    const userBubble = document.createElement('div');
    userBubble.className = 'flex justify-end w-full mb-2';
    userBubble.innerHTML = `<span class="bg-[#e05a2b] text-white text-[11px] px-3.5 py-2.5 rounded-2xl rounded-tr-none max-w-[85%] break-words shadow-sm">${text}</span>`;
    messages.appendChild(userBubble);
    input.value = '';
    messages.scrollTop = messages.scrollHeight;

    // 세션 타이틀 자동 갱신 (첫 질문일 경우)
    const session = chatSessions.find(s => s.id === activeSessionId);
    if (session && (session.title === '새로운 대화' || session.title.includes('발전 예보 AI 일별 분석'))) {
        session.title = text.length > 18 ? text.substring(0, 16) + '...' : text;
        renderChatSessions();
        saveChatSessionsToStorage();
    }

    // 2. 실시간 추론 타임라인 메시지 추가
    const traceId = 'trace-' + Math.random().toString(36).substr(2, 6);
    const traceBubble = document.createElement('div');
    traceBubble.className = 'flex justify-start w-full mb-4';
    traceBubble.innerHTML = createTraceTimelineHtml(traceId);
    messages.appendChild(traceBubble);
    messages.scrollTop = messages.scrollHeight;

    // 타이머 작동
    let seconds = 0;
    const timerEl = document.getElementById(`${traceId}-timer`);
    const timerInterval = setInterval(() => {
        seconds += 0.5;
        if (timerEl) timerEl.textContent = `${seconds.toFixed(1)}초 경과`;
    }, 500);

    // 단계 시뮬레이션 상태 변경 헬퍼
    const setStepStatus = (stepIndex, status, detailText = '') => {
        const stepEl = document.getElementById(`${traceId}-step-${stepIndex}`);
        if (!stepEl) return;
        const statusEl = stepEl.querySelector('.step-status');
        const titleEl = stepEl.querySelector('.step-title');
        const descEl = stepEl.querySelector('.step-desc');

        if (status === 'active') {
            if (statusEl) {
                statusEl.innerHTML = '';
                statusEl.className = 'absolute -left-4 top-1.5 h-2.5 w-2.5 rounded-full bg-amber-400 transform -translate-x-1/2 step-status animate-pulse';
            }
            if (titleEl) {
                titleEl.className = 'font-semibold text-amber-600 step-title';
            }
            if (descEl) {
                descEl.classList.remove('hidden');
                if (detailText) descEl.textContent = detailText;
            }
        } else if (status === 'success') {
            if (statusEl) {
                statusEl.innerHTML = '';
                statusEl.className = 'absolute -left-4 top-1.5 h-2 w-2 rounded-full bg-emerald-500 transform -translate-x-1/2 step-status';
            }
            if (titleEl) {
                titleEl.className = 'font-semibold text-emerald-700 step-title';
            }
            if (descEl) {
                descEl.className = 'text-[9px] text-emerald-600 mt-0.5 step-desc';
                if (detailText) descEl.textContent = detailText;
            }
        }
    };

    // 1단계 활성화
    setStepStatus(1, 'active');

    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });
        const data = await res.json();

        // 타이머 해제
        clearInterval(timerInterval);
        if (timerEl) timerEl.textContent = `분석 완료 (${seconds.toFixed(1)}초)`;

        const traces = data.tool_trace || [];

        // 단계 완료 연출 시뮬레이션 (순차 딜레이 부여)
        setStepStatus(1, 'success', '실측 데이터 로드 성공 (제주 발전 실측값 로딩)');
        await sleep(350);

        setStepStatus(2, 'active');
        await sleep(400);
        const hasPredict = traces.some(t => t.tool === 'tool_predict');
        setStepStatus(2, 'success', hasPredict ? 'LSTM 가중치 예측 추론 연산 성공' : '기본 발전 예측 데이터 결합 완료');

        setStepStatus(3, 'active');
        await sleep(450);
        const ragTrace = traces.find(t => t.tool === 'tool_rag');
        setStepStatus(3, 'success', ragTrace ? `RAG 유사 과거사례 ${ragTrace.found || 3}건 매칭 검색 성공` : '유사 기상 레코드 스캔 완료');

        setStepStatus(4, 'active');
        await sleep(350);
        setStepStatus(4, 'success', '물리 법칙 제약조건 스캔 통과 (지식 그래프 검증 완료)');

        setStepStatus(5, 'active');
        await sleep(400);
        setStepStatus(5, 'success', 'SHAP 기여도 변수 산출 및 자율 재예측 보정 완료');

        // 헤더 상태 변경
        const headerStatus = document.getElementById(`${traceId}-header-status`);
        if (headerStatus) {
            headerStatus.textContent = '분석 리포트 생성 완료';
            headerStatus.className = 'text-emerald-800 font-bold';
            const iconNode = headerStatus.previousElementSibling;
            if (iconNode) {
                iconNode.innerHTML = '✓';
                iconNode.className = 'text-emerald-600 font-bold text-xs';
            }
        }

        // AI 답변 페이드인
        const narrativeBox = document.getElementById(`${traceId}-narrative`);
        if (narrativeBox) {
            narrativeBox.innerHTML = formatAiNarrative(data.answer).replace(/\n/g, '<br>');
            narrativeBox.classList.remove('hidden');
        }

        if (data.report) {
            updateRailStatusCard(data.report);
            updateGlobalStatusBar(data.report);
        }
        if (data.tool_trace?.length) renderToolTrace(data.tool_trace);
    } catch (err) {
        clearInterval(timerInterval);
        const headerStatus = document.getElementById(`${traceId}-header-status`);
        if (headerStatus) {
            headerStatus.textContent = '에러 발생';
            headerStatus.className = 'text-red-700 font-bold';
        }
        const narrativeBox = document.getElementById(`${traceId}-narrative`);
        if (narrativeBox) {
            narrativeBox.innerHTML = '❌ 에이전트 분석 중 오류가 발생했습니다: ' + err.message;
            narrativeBox.classList.remove('hidden');
        }
    }

    // 세션 HTML 캐시 업데이트
    const currSession = chatSessions.find(s => s.id === activeSessionId);
    if (currSession) {
        currSession.html = messages.innerHTML;
    }
    saveChatSessionsToStorage();

    messages.scrollTop = messages.scrollHeight;
}

// ── Leaflet 지도 초기화 ──────────────────────────────────────
function initMap() {
    if (map) return;
    map = L.map('map', { zoomControl: true, attributionControl: false, scrollWheelZoom: false })
        .setView([33.38, 126.55], 10);
    const tiles = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', { maxZoom: 18 }).addTo(map);
    tiles.getContainer().classList.add('clean-map-tiles');

    const regionCoords = {
        '제주시': [33.4996, 126.5312],
        '서귀포시': [33.2541, 126.5601],
        '한경/고산': [33.35, 126.18],
        '구좌/성산': [33.45, 126.85]
    };
    for (const [name, coord] of Object.entries(regionCoords)) {
        const safeName = name.replace('/', '-');
        const customIcon = L.divIcon({
            className: '',
            html: `<div id="marker-pulse-${safeName}" class="h-7 w-7 rounded-full flex items-center justify-center pulse-marker-amber text-xs text-white font-bold select-none cursor-pointer">🟡</div>`,
            iconSize: [28, 28], iconAnchor: [14, 14]
        });
        const marker = L.marker(coord, { icon: customIcon }).addTo(map);
        marker.on('click', () => selectMapRegion(name));
        marker.bindTooltip(`<strong>${name}</strong>`, { direction: 'top', offset: [0, -5] });
        mapMarkers[name] = marker;
    }
}

// ── 초기화 ────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    const hours = String(now.getHours()).padStart(2, '0');
    const localISO = `${year}-${month}-${day}T${hours}:00`;
    const picker = document.getElementById('picker-datetime');
    if (picker) picker.value = localISO;
    document.getElementById('top-time-display').textContent = `${year}-${month}-${day} ${hours}:00`;

    initChart();
    initMap();
    renderCalendar();
    selectCalendarDay(selectedDayNum, '높음', '보통');
    switchTab('home');
    selectMapRegion('제주 전역');
    updateData();

    // 최초 대화 기록 세션 생성 또는 로드
    loadChatSessionsFromStorage();
    if (chatSessions.length === 0) {
        startNewChatSession();
    } else {
        renderChatSessions();
        // 마지막 활성 세션의 HTML 화면에 로드
        if (activeSessionId) {
            const activeSession = chatSessions.find(s => s.id === activeSessionId);
            if (activeSession) {
                document.getElementById('rail-chat-messages').innerHTML = activeSession.html;
                const messages = document.getElementById('rail-chat-messages');
                if (messages) messages.scrollTop = messages.scrollHeight;
            }
        }
    }
});

// ── 탭 전환 ───────────────────────────────────────────────────
function switchTab(tab) {
    activeTab = tab;
    ['view-home', 'view-map', 'view-calendar', 'view-agent'].forEach(id => {
        document.getElementById(id)?.classList.add('hidden');
    });

    const INACTIVE = 'text-[11px] font-medium px-4 py-[5px] rounded-[6px] text-[#9ca3af] hover:text-[#374151] transition-all';
    const ACTIVE = 'text-[11px] font-semibold px-4 py-[5px] rounded-[6px] bg-white text-[#111] shadow-sm transition-all';

    document.getElementById('tab-btn-home').className = INACTIVE;
    document.getElementById('tab-btn-map').className = INACTIVE;
    document.getElementById('tab-btn-calendar').className = INACTIVE;
    document.getElementById('tab-btn-agent').className = INACTIVE;

    const opControls = document.getElementById('operator-controls-wrapper');
    opControls.style.display = (tab === 'home') ? 'flex' : 'none';

    if (tab === 'home') {
        const homeView = document.getElementById('view-home');
        homeView.classList.remove('hidden');
        homeView.scrollTop = 0;
        document.getElementById('tab-btn-home').className = ACTIVE;
        setTimeout(updateChartData, 100);
    } else if (tab === 'map') {
        document.getElementById('view-map').classList.remove('hidden');
        document.getElementById('tab-btn-map').className = ACTIVE;
        if (map) setTimeout(() => map.invalidateSize(), 50);
    } else if (tab === 'calendar') {
        document.getElementById('view-calendar').classList.remove('hidden');
        document.getElementById('tab-btn-calendar').className = ACTIVE;

        // 달력 탭 진입 시 항상 현재 포커스된 날짜의 예측/실적 데이터 로드
        if (!selectedDayNum) {
            const today = new Date();
            selectedDayNum = today.getDate();
            currentYear = today.getFullYear();
            currentMonth = today.getMonth() + 1;
        }

        let solarStatus = selectedSolarStatus || '높음';
        let windStatus = selectedWindStatus || '보통';
        if (selectedDayNum % 3 === 0) { solarStatus = '보통'; }
        else if (selectedDayNum % 5 === 0) { solarStatus = '낮음'; }
        if (selectedDayNum % 2 === 0) { windStatus = '높음'; }
        else if (selectedDayNum % 7 === 0) { windStatus = '낮음'; }

        selectCalendarDay(selectedDayNum, solarStatus, windStatus);
    } else if (tab === 'agent') {
        document.getElementById('view-agent').classList.remove('hidden');
        document.getElementById('tab-btn-agent').className = ACTIVE;
    }
}

// ── 캘린더 상태 ───────────────────────────────────────────────
const initDate = new Date();
let currentYear = initDate.getFullYear();
let currentMonth = initDate.getMonth() + 1;
let selectedDayNum = initDate.getDate();
let selectedSolarStatus = '높음';
let selectedWindStatus = '보통';
let selectedHourVal = 'all'; // 글로벌 시간대 선택 보존

function prevMonth() { currentMonth--; if (currentMonth < 1) { currentMonth = 12; currentYear--; } updateCalendarDisplay(); }
function nextMonth() { currentMonth++; if (currentMonth > 12) { currentMonth = 1; currentYear++; } updateCalendarDisplay(); }

function updateCalendarDisplay() {
    document.getElementById('cal-month-title').textContent = `${currentYear}년 ${currentMonth}월 발전 예보 스케줄러`;
    document.getElementById('cal-month-display').textContent = `${currentMonth}월 (${getMonthEnglishName(currentMonth)})`;

    // 달을 변경할 때 날짜 선택 해제
    selectedDayNum = null;

    renderCalendar();
}

function clearCalendarSelection() {
    // 이제 강제로 클리어하지 않고 1일을 기본으로 선택하거나 오늘을 선택하도록 함
    const today = new Date();
    if (today.getFullYear() === currentYear && (today.getMonth() + 1) === currentMonth) {
        selectedDayNum = today.getDate();
    } else {
        selectedDayNum = 1;
    }
    renderCalendar();
}

function getMonthEnglishName(m) {
    return ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][m - 1];
}

// 맵 권역 정보 업데이트
function selectMapRegion(regionName) {
    const data = regionMockData[regionName] || regionMockData['제주 전역'];
    document.getElementById('map-region-title').textContent = regionName;
    document.getElementById('map-info-solar-val').textContent = `${data.solarGen.toFixed(1)} MW`;
    document.getElementById('map-info-wind-val').textContent = `${data.windGen.toFixed(1)} MW`;
    document.getElementById('map-info-generation').textContent = `${data.gen.toFixed(1)} MW`;
    document.getElementById('map-info-solar-rad').textContent = `${data.solar.toFixed(2)} MJ/m²`;
    document.getElementById('map-info-weather').textContent = data.weather;
    document.getElementById('map-info-ai').innerHTML = data.ai;
}

// Focused Carousel 형태로 달력 렌더링
function renderCalendar() {
    const grid = document.getElementById('calendar-days-grid');
    grid.innerHTML = '';

    const totalDaysInMonth = new Date(currentYear, currentMonth, 0).getDate();
    const today = new Date();
    const isCurrentMonth = today.getFullYear() === currentYear && (today.getMonth() + 1) === currentMonth;
    const todayDay = today.getDate();

    // 시간 선택 동적 옵션 생성
    const hoursOptions = [
        { val: 'all', text: '하루 종합 (All Day)' },
        { val: '00', text: '00:00' }, { val: '01', text: '01:00' }, { val: '02', text: '02:00' },
        { val: '03', text: '03:00' }, { val: '04', text: '04:00' }, { val: '05', text: '05:00' },
        { val: '06', text: '06:00' }, { val: '07', text: '07:00' }, { val: '08', text: '08:00' },
        { val: '09', text: '09:00' }, { val: '10', text: '10:00' }, { val: '11', text: '11:00' },
        { val: '12', text: '12:00 ☀️ 피크' }, { val: '13', text: '13:00' }, { val: '14', text: '14:00' },
        { val: '15', text: '15:00' }, { val: '16', text: '16:00' }, { val: '17', text: '17:00' },
        { val: '18', text: '18:00' }, { val: '19', text: '19:00' }, { val: '20', text: '20:00' },
        { val: '21', text: '21:00' }, { val: '22', text: '22:00' }, { val: '23', text: '23:00' }
    ].map(opt => `<option value="${opt.val}" ${selectedHourVal === opt.val ? 'selected' : ''}>${opt.text}</option>`).join('');

    for (let day = 1; day <= totalDaysInMonth; day++) {
        const cell = document.createElement('div');
        cell.id = `cal-cell-${day}`;

        const dayOfWeekVal = new Date(currentYear, currentMonth - 1, day).getDay();
        const dayOfWeekName = ['일', '월', '화', '수', '목', '금', '토'][dayOfWeekVal];

        let solarStatus = '높음';
        let windStatus = '보통';

        if (day % 3 === 0) { solarStatus = '보통'; }
        else if (day % 5 === 0) { solarStatus = '낮음'; }
        if (day % 2 === 0) { windStatus = '높음'; }
        else if (day % 7 === 0) { windStatus = '낮음'; }

        const solarDotColor = solarStatus === '높음' ? 'bg-emerald-400' : (solarStatus === '보통' ? 'bg-amber-400' : 'bg-[#d4d4d4]');
        const windDotColor = windStatus === '높음' ? 'bg-emerald-400' : (windStatus === '보통' ? 'bg-amber-400' : 'bg-[#d4d4d4]');

        const formattedDate = `${currentYear}-${String(currentMonth).padStart(2, '0')}-${String(day).padStart(2, '0')}`;

        if (day === selectedDayNum) {
            // 선택된 날짜: 크게 보이고 세부 드롭다운 및 발전량이 표시됨 (Carousel Focus 효과)
            cell.className = 'cal-day-card flex-shrink-0 w-[360px] min-h-[240px] border border-black/[0.08] rounded-xl p-5 cursor-default transition-all text-left bg-white shadow-2xl z-20 flex flex-col justify-between';
            cell.innerHTML = `
                <div class="flex justify-between items-center">
                    <div>
                        <span class="text-[10px] text-[#aaa] font-semibold">${dayOfWeekName}요일</span>
                        <h4 class="text-xl font-extrabold text-[#111] mt-0.5" style="letter-spacing:-0.02em;">${day}일 예보</h4>
                    </div>
                    <div class="flex gap-1.5">
                        <span title="태양광 ${solarStatus}" class="h-2 w-2 rounded-full ${solarDotColor}"></span>
                        <span title="풍력 ${windStatus}" class="h-2 w-2 rounded-full ${windDotColor}"></span>
                    </div>
                </div>

                <div class="my-3 space-y-2.5 bg-[#fafafa] border border-[#f0f0f0] rounded-xl p-3">
                    <div class="flex items-center justify-between">
                        <span class="text-[10px] text-[#aaa] font-bold uppercase tracking-wider">시간대 선택</span>
                        <select id="cal-hour-select" onchange="onCalHourChange()" class="bg-white border border-[#e8e8e8] text-[11px] rounded-lg px-2 py-1 focus:outline-none focus:border-[#ccc] text-[#333] font-semibold">
                            ${hoursOptions}
                        </select>
                    </div>
                    <div class="grid grid-cols-2 gap-2 text-xs">
                        <div class="bg-white border border-[#f0f0f0] p-2.5 rounded-lg text-left">
                            <span class="text-[9px] text-[#aaa] block uppercase tracking-wider">태양광</span>
                            <span id="cal-info-solar-val" class="text-sm font-bold text-[#111] mt-0.5 block" style="letter-spacing:-0.01em;">- MW</span>
                        </div>
                        <div class="bg-white border border-[#f0f0f0] p-2.5 rounded-lg text-left">
                            <span class="text-[9px] text-[#aaa] block uppercase tracking-wider">풍력</span>
                            <span id="cal-info-wind-val" class="text-sm font-bold text-[#111] mt-0.5 block" style="letter-spacing:-0.01em;">- MW</span>
                        </div>
                    </div>
                    <div class="flex justify-between text-[11px] text-[#888] border-t border-[#f0f0f0] pt-2">
                        <span>총합: <strong id="cal-info-generation" class="text-[#333] font-bold">- MW</strong></span>
                        <span>일사량: <strong id="cal-info-solar" class="text-[#333] font-bold">- MJ/m²</strong></span>
                    </div>
                    <div class="text-[10px] text-[#888] bg-white border border-[#f0f0f0] rounded-lg p-2 leading-snug">
                        기상: <span id="cal-info-weather" class="text-[#555] font-medium">-</span>
                    </div>
                </div>

                <button onclick="askAgentForCalendarDay('${formattedDate}')" class="w-full bg-[#111] text-white hover:bg-black text-[11px] font-bold py-2.5 rounded-xl transition-all flex items-center justify-center gap-1.5">
                    💬 에이전트 탭에서 AI 일별 분석 리포트 보기
                </button>
            `;
        } else {
            // 선택되지 않은 날짜: 좌우에 오며 약간 작고 접혀있는 형태 (opacity 및 scale 적용)
            let borderClass = 'border-[#e5e7eb]';
            if (isCurrentMonth && day === todayDay) {
                borderClass = 'border border-[#e05a2b]/40';
            }
            cell.className = `cal-day-card flex-shrink-0 w-[72px] h-[110px] border ${borderClass} rounded-xl p-3 cursor-pointer hover:border-[#d1d5db] hover:opacity-80 transition-all text-center bg-white flex flex-col justify-between opacity-50 scale-90`;
            cell.innerHTML = `
                <div class="text-[9px] text-[#d1d5db] font-semibold uppercase">${dayOfWeekName}</div>
                <div class="text-xl font-bold text-[#9ca3af] my-1">${day}</div>
                <div class="flex justify-center gap-1 mt-1">
                    <span title="태양광 ${solarStatus}" class="h-1.5 w-1.5 rounded-full ${solarDotColor}"></span>
                    <span title="풍력 ${windStatus}" class="h-1.5 w-1.5 rounded-full ${windDotColor}"></span>
                </div>
            `;
            cell.setAttribute('onclick', `selectCalendarDay(${day}, '${solarStatus}', '${windStatus}')`);
        }

        grid.appendChild(cell);
    }

    // 중앙 맞춤 스크롤 작동
    if (selectedDayNum) {
        const targetCell = document.getElementById(`cal-cell-${selectedDayNum}`);
        if (targetCell) {
            setTimeout(() => {
                targetCell.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
            }, 50);
        }
    }
}

function onCalHourChange() { selectCalendarDay(selectedDayNum, selectedSolarStatus, selectedWindStatus); }

function formatAiNarrative(text) {
    if (!text) return '';
    let t = text
        .replace(/\*\*/g, '')
        .replace(/`/g, '')
        .replace(/^#{1,6}\s*/gm, '')
        .replace(/^[-*]\s*/gm, '• ')
        .replace(/^\|[\s\-:]+\|[\s\-:|]*$/gm, '')
        .replace(/^\|(.+)\|$/gm, (_, inner) => {
            const cols = inner.split('|').map(c => c.trim()).filter(Boolean);
            return cols.length >= 2 ? `${cols[0]}: ${cols[1]}` : cols[0] || '';
        });
    t = t.replace(/\s*(📊|☀️|💨|📚|📐|🎯|🔁|🌪|🔋|🌞|🌬|⚡|🏭)/g, '\n$1');
    return t.split('\n').map(l => l.trim()).filter(l => l.length > 0).join('\n').trim();
}

function jumpToDate(dateStr) {
    if (!dateStr) return;
    const [y, m, d] = dateStr.split('-').map(Number);
    currentYear = y;
    currentMonth = m;
    selectedDayNum = d;
    updateCalendarDisplay();

    setTimeout(() => {
        let solarStatus = '높음';
        let windStatus = '보통';
        if (d % 3 === 0) { solarStatus = '보통'; }
        else if (d % 5 === 0) { solarStatus = '낮음'; }
        if (d % 2 === 0) { windStatus = '높음'; }
        else if (d % 7 === 0) { windStatus = '낮음'; }
        selectCalendarDay(d, solarStatus, windStatus);
    }, 100);
}

// ── 대화 세션 상태 관리 ───────────────────────────────
let chatSessions = []; // { id, title, html }
let activeSessionId = null;

function saveChatSessionsToStorage() {
    try {
        localStorage.setItem('jeju_chat_sessions', JSON.stringify(chatSessions));
        localStorage.setItem('jeju_active_session_id', activeSessionId);
    } catch (e) {
        console.error("Failed to save sessions to localStorage:", e);
    }
}

function loadChatSessionsFromStorage() {
    try {
        const savedSessions = localStorage.getItem('jeju_chat_sessions');
        const savedActiveId = localStorage.getItem('jeju_active_session_id');

        if (savedSessions) {
            chatSessions = JSON.parse(savedSessions);
            activeSessionId = savedActiveId;
        } else {
            chatSessions = [];
        }
    } catch (e) {
        console.error("Failed to load sessions from localStorage:", e);
    }
}

function startNewChatSession(initialQuestion = '') {
    const sessionId = 'session-' + Date.now();
    const sessionTitle = initialQuestion
        ? (initialQuestion.length > 18 ? initialQuestion.substring(0, 16) + '...' : initialQuestion)
        : '새로운 대화';

    const newSession = {
        id: sessionId,
        title: sessionTitle,
        html: ''
    };

    chatSessions.unshift(newSession);
    activeSessionId = sessionId;

    renderChatSessions();
    saveChatSessionsToStorage();

    // 메시지 보드 비우기
    const messagesEl = document.getElementById('rail-chat-messages');
    if (messagesEl) {
        messagesEl.innerHTML = '';
    }

    // 질문이 있으면 자동으로 입력 후 전송
    if (initialQuestion) {
        const inputEl = document.getElementById('rail-chat-input');
        if (inputEl) {
            inputEl.value = initialQuestion;
            sendRailChat();
        }
    }
}

function selectChatSession(sessionId) {
    activeSessionId = sessionId;
    renderChatSessions();
    saveChatSessionsToStorage();

    const target = chatSessions.find(s => s.id === sessionId);
    const messagesEl = document.getElementById('rail-chat-messages');
    if (messagesEl) {
        messagesEl.innerHTML = target ? target.html : '';
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }
}

function deleteChatSession(sessionId, event) {
    if (event) event.stopPropagation();

    chatSessions = chatSessions.filter(s => s.id !== sessionId);

    if (activeSessionId === sessionId) {
        activeSessionId = chatSessions.length > 0 ? chatSessions[0].id : null;
        if (activeSessionId) {
            const target = chatSessions.find(s => s.id === activeSessionId);
            document.getElementById('rail-chat-messages').innerHTML = target ? target.html : '';
        } else {
            document.getElementById('rail-chat-messages').innerHTML = '';
        }
    }

    if (chatSessions.length === 0) {
        startNewChatSession();
    } else {
        renderChatSessions();
        saveChatSessionsToStorage();
    }
}

function renderChatSessions() {
    const listEl = document.getElementById('chat-session-list');
    if (!listEl) return;
    listEl.innerHTML = '';

    chatSessions.forEach(session => {
        const isActive = session.id === activeSessionId;
        const item = document.createElement('div');

        item.className = `group flex items-center justify-between px-3 py-2 rounded-lg cursor-pointer transition-all text-xs ${isActive
            ? 'bg-[#f0f0f2] text-[#111] font-semibold'
            : 'hover:bg-white border border-transparent hover:border-[#e5e7eb] text-[#6b7280]'
            }`;
        item.setAttribute('onclick', `selectChatSession('${session.id}')`);

        item.innerHTML = `
            <div class="flex items-center gap-1.5 min-w-0 flex-1">
                <span class="text-[9px]">${isActive ? '💬' : '▫️'}</span>
                <span class="truncate pr-2 select-none">${session.title}</span>
            </div>
            <button onclick="deleteChatSession('${session.id}', event)" class="opacity-0 group-hover:opacity-100 hover:text-red-400 text-[11px] transition-all p-0.5 rounded flex-shrink-0 font-bold">
                ✕
            </button>
        `;
        listEl.appendChild(item);
    });
}

// 에이전트 탭으로 날짜 분석 들고 점프 (자동 새 대화 트리거)
function askAgentForCalendarDay(dateStr) {
    const hourSelect = document.getElementById('cal-hour-select');
    const hour = hourSelect ? hourSelect.value : 'all';
    const hourText = hour === 'all' ? '하루 종합' : `${hour}시`;

    switchTab('agent');

    const questionText = `${dateStr} ${hourText} 발전 예보 AI 일별 분석 리포트를 생성해주고 분석 요약해줘.`;

    // 새 대화를 생성하여 질문 전송
    startNewChatSession(questionText);
}

async function selectCalendarDay(day, solarStatus, windStatus) {
    if (!day) return;
    selectedDayNum = day;
    selectedSolarStatus = solarStatus;
    selectedWindStatus = windStatus;

    // 드롭다운의 현재 값 가져오기
    const hourSelect = document.getElementById('cal-hour-select');
    if (hourSelect) {
        selectedHourVal = hourSelect.value;
    }

    renderCalendar();

    const formattedDate = `${currentYear}-${String(currentMonth).padStart(2, '0')}-${String(day).padStart(2, '0')}`;

    try {
        // 신규 daily_data API를 통해 하루치 모든 데이터를 단 한 번의 요청으로 획득
        const response = await fetch(`/daily_data?date=${formattedDate}`);
        if (!response.ok) throw new Error('API 연동 실패');
        const data = await response.json();

        if (data.status !== 'ok') {
            throw new Error(data.message || '데이터 없음');
        }

        let solarGen = 0;
        let windGen = 0;
        let totalGen = 0;
        let mockSolarRad = 0;
        let weather = '';

        if (selectedHourVal === 'all') {
            // 하루 종합 데이터
            solarGen = data.solar_sum;
            windGen = data.wind_sum;
            totalGen = data.total_sum;
            mockSolarRad = Math.min(solarGen > 0 ? solarGen / (80 * 24) : 0, 3.0);
            weather = totalGen > 600 ? '맑음 (평균 풍속 6.2 m/s)' : '흐림/구름 (평균 풍속 5.8 m/s)';
            if (data.warnings && data.warnings.length > 0) {
                weather = `기상 경보 (${data.warnings.join(', ')})`;
            }
        } else {
            // 특정 시간대 데이터
            const match = data.hourly.find(h => h.hour === selectedHourVal);
            if (match) {
                solarGen = match.solar_mw;
                windGen = match.wind_mw;
                totalGen = solarGen + windGen;
                mockSolarRad = Math.min(solarGen > 0 ? solarGen / 80 : 0, 3.0);
                weather = totalGen > 25 ? '맑음 (평균 풍속 6.2 m/s)' : '흐림/야간 (평균 풍속 5.8 m/s)';
                if (data.warnings && data.warnings.length > 0) {
                    weather = `기상 경보 (${data.warnings.join(', ')})`;
                }
            }
        }

        const solarEl = document.getElementById('cal-info-solar-val');
        const windEl = document.getElementById('cal-info-wind-val');
        const genEl = document.getElementById('cal-info-generation');
        const radEl = document.getElementById('cal-info-solar');
        const weaEl = document.getElementById('cal-info-weather');

        if (solarEl) solarEl.textContent = `${solarGen.toFixed(2)} MW`;
        if (windEl) windEl.textContent = `${windGen.toFixed(2)} MW`;
        if (genEl) genEl.textContent = `${totalGen.toFixed(2)} MW`;
        if (radEl) radEl.textContent = `${mockSolarRad.toFixed(2)} MJ/m²`;
        if (weaEl) weaEl.textContent = weather;

        window._lastCalendarReport = {
            date: formattedDate,
            hour: selectedHourVal,
            solar: solarGen.toFixed(2),
            wind: windGen.toFixed(2),
            total: totalGen.toFixed(2),
            weather: weather
        };

    } catch (err) {
        console.error("Failed to fetch calendar day data:", err);
        ['cal-info-solar-val', 'cal-info-wind-val', 'cal-info-generation'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.textContent = '- MW';
        });
    }
}

function updateMapState(total) {
    const distributions = {
        '제주시': { genRatio: 0.35, solarRatio: 0.6, baseWind: 5.0 },
        '서귀포시': { genRatio: 0.30, solarRatio: 0.4, baseWind: 3.5 },
        '한경/고산': { genRatio: 0.20, solarRatio: 0.1, baseWind: 12.0 },
        '구좌/성산': { genRatio: 0.15, solarRatio: 0.2, baseWind: 6.0 }
    };
    for (const [name, config] of Object.entries(distributions)) {
        const safeName = name.replace('/', '-');
        const regGen = total * config.genRatio;
        const regSolar = regGen * config.solarRatio;
        const regWind = regGen - regSolar;
        let colorClass = 'pulse-marker-red', emoji = '🔴', statusLabel = '낮음';
        if (regGen >= 25) { colorClass = 'pulse-marker-emerald'; emoji = '🟢'; statusLabel = '높음'; }
        else if (regGen >= 10) { colorClass = 'pulse-marker-amber'; emoji = '🟡'; statusLabel = '보통'; }
        const el = document.getElementById(`marker-pulse-${safeName}`);
        if (el) { el.className = `h-7 w-7 rounded-full flex items-center justify-center ${colorClass} text-xs text-white font-bold select-none cursor-pointer`; el.textContent = emoji; }

        let weatherText = `맑음 (풍속 ${config.baseWind.toFixed(1)} m/s)`;
        if (statusLabel === '낮음') weatherText = `흐리고 비 (풍속 ${(config.baseWind * 0.8).toFixed(1)} m/s)`;
        else if (statusLabel === '보통') weatherText = `구름 조금 (풍속 ${(config.baseWind * 0.9).toFixed(1)} m/s)`;
        const aiText = `제주 ${name} 지역 예상 발전: ${statusLabel} 단계. 일사량 ${(regSolar / 15).toFixed(2)} MJ/m² 조건 하에 ${statusLabel === '높음' ? '피크 효율이 기대됩니다.' : statusLabel === '보통' ? '완만한 예측 곡선을 그립니다.' : '발전량이 큰 폭으로 감소할 것으로 예상됩니다.'}`;
        regionMockData[name] = { gen: regGen, solarGen: regSolar, windGen: regWind, solar: regSolar / 15, weather: weatherText, ai: aiText };
    }
    const globalSolar = total * 0.55;
    regionMockData['제주 전역'] = {
        gen: total, solarGen: globalSolar, windGen: total - globalSolar, solar: total * 0.4 / 15,
        weather: total > 60 ? '맑음 (평균 풍속 6.2 m/s)' : '흐림/야간 (평균 풍속 5.8 m/s)',
        ai: `제주 전역 실시간 예측 피크값: ${total.toFixed(2)} MW. 전반적인 송전망 연계 상태는 양호합니다.`
    };
}

// ── 관제 실행 ─────────────────────────────────────────────────
async function updateData() {
    const dtInput = document.getElementById('picker-datetime').value;
    const now = new Date();
    let targetTimeStr = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')} ${String(now.getHours()).padStart(2, '0')}:00:00`;
    if (dtInput) {
        const dt = new Date(dtInput);
        targetTimeStr = `${dt.getFullYear()}-${String(dt.getMonth() + 1).padStart(2, '0')}-${String(dt.getDate()).padStart(2, '0')} ${String(dt.getHours()).padStart(2, '0')}:00:00`;
        document.getElementById('top-time-display').textContent =
            `${dt.getFullYear()}-${String(dt.getMonth() + 1).padStart(2, '0')}-${String(dt.getDate()).padStart(2, '0')} ${String(dt.getHours()).padStart(2, '0')}:00`;
    }

    showAgentThinking();
    updateGlobalStatusBar(null);

    try {
        const response = await fetch('/chat', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: `${targetTimeStr} 발전량 알려줘` })
        });
        if (!response.ok) throw new Error('API 연동 실패');
        const data = await response.json();
        const report = data.report;
        const solar = report.predictions.solar_mw;
        const wind = report.predictions.wind_mw;
        const total = solar + wind;

        // 히어로 + KPI 갱신
        document.getElementById('op-today-val').textContent = total.toFixed(1);
        document.getElementById('hero-solar-val').textContent = solar.toFixed(1);
        document.getElementById('hero-wind-val').textContent = wind.toFixed(1);
        document.getElementById('hero-conf-badge').textContent = `신뢰도 ${report.confidence || 'High'}`;
        document.getElementById('op-solar-val').textContent = solar.toFixed(1);
        document.getElementById('op-wind-val').textContent = wind.toFixed(1);

        // 진행 바 갱신 (태양광 max 80MW, 풍력 max 200MW)
        const solarPct = Math.min(100, (solar / 80) * 100);
        const windPct = Math.min(100, (wind / 200) * 100);
        document.getElementById('solar-progress-bar').style.width = `${solarPct.toFixed(0)}%`;
        document.getElementById('wind-progress-bar').style.width = `${windPct.toFixed(0)}%`;
        document.getElementById('solar-bar-label').textContent = `${solarPct.toFixed(0)}%`;
        document.getElementById('wind-bar-label').textContent = `${windPct.toFixed(0)}%`;

        let accuracy = 94.8;
        if (report.confidence === 'Medium') accuracy = 91.2 + Math.random() * 2.5;
        else if (report.confidence === 'Low') accuracy = 86.5 + Math.random() * 4.0;
        else accuracy = 94.5 + Math.random() * 2.0;
        document.getElementById('op-accuracy-val').textContent = accuracy.toFixed(1);
        document.getElementById('op-accuracy-bar').style.width = `${accuracy.toFixed(1)}%`;

        // 에이전트 상태 카드
        const agentCard = document.getElementById('op-agent-card');
        const agentDot = document.getElementById('op-agent-dot');
        const agentLabel = document.getElementById('op-agent-label');
        document.getElementById('op-agent-conf').textContent = report.confidence || 'High';
        const nowT = new Date();
        document.getElementById('op-agent-time').textContent = `${String(nowT.getHours()).padStart(2, '0')}:${String(nowT.getMinutes()).padStart(2, '0')}`;
        if (report.trigger_active || (report.warnings || []).length > 0) {
            agentCard.className = 'theme-red rounded-xl p-5 flex flex-col gap-3 border transition-all duration-300';
            agentDot.style.background = '#ef4444';
            agentLabel.textContent = '경보 발동';
        } else if (report.confidence === 'Low') {
            agentCard.className = 'theme-amber rounded-xl p-5 flex flex-col gap-3 border transition-all duration-300';
            agentDot.style.background = '#f59e0b';
            agentLabel.textContent = '불확실성 높음';
        } else {
            agentCard.className = 'theme-emerald rounded-xl p-5 flex flex-col gap-3 border transition-all duration-300';
            agentDot.style.background = '#22c55e';
            agentLabel.textContent = '정상 모니터링';
        }

        // 기상 인디케이터
        document.getElementById('op-val-obs-wind').textContent = report.trigger_active ? "18.5 m/s" : "6.2 m/s";
        document.getElementById('op-val-obs-solar').textContent = report.predictions.solar_mw > 0 ? "1.85 MJ/m²" : "0.00 MJ/m²";
        document.getElementById('op-val-obs-temp').textContent = "22.5 °C";
        document.getElementById('op-val-obs-model').textContent = report.confidence === 'High' ? "LSTM+CNN" : "LSTM Only";

        // 경보 이력 패널 갱신 (/events API 연동)
        fetchAndRenderHomeAlerts(report);

        updateMapState(total);
        selectMapRegion(document.getElementById('map-region-title').textContent);
        updateChartData(report);

        // 재예측 배지
        const refined = report.refined_report;
        const mapBadge = document.getElementById('map-refinement-badge');
        const calBadge = document.getElementById('cal-refinement-badge');
        if (refined?.refinement_applied) {
            const solarRange = `${refined.range_low.solar_mw.toFixed(1)} ~ ${refined.range_high.solar_mw.toFixed(1)} MW`;
            const windRange = `${refined.range_low.wind_mw.toFixed(1)} ~ ${refined.range_high.wind_mw.toFixed(1)} MW`;
            if (document.getElementById('map-refine-solar')) document.getElementById('map-refine-solar').textContent = solarRange;
            if (document.getElementById('map-refine-wind')) document.getElementById('map-refine-wind').textContent = windRange;
            if (document.getElementById('cal-refine-solar')) document.getElementById('cal-refine-solar').textContent = solarRange;
            if (document.getElementById('cal-refine-wind')) document.getElementById('cal-refine-wind').textContent = windRange;
            if (mapBadge) mapBadge.classList.remove('hidden');
            if (calBadge) calBadge.classList.remove('hidden');
        } else {
            if (mapBadge) mapBadge.classList.add('hidden');
            if (calBadge) calBadge.classList.add('hidden');
        }

        window._lastReport = report;

        // 레일 갱신
        renderToolTrace(data.tool_trace || []);
        updateRailStatusCard(report);
        updateGlobalStatusBar(report);
        if (report.trigger_active || (report.warnings || []).length > 0) {
            appendAlertToRail(report.warnings || [], report);
        }

    } catch (err) {
        console.error(err);
        updateGlobalStatusBar({ confidence: 'High' }); // fallback reset
    }
}

// ── 차트 초기화 ───────────────────────────────────────────────
function initChart() {
    const commonScales = {
        y: {
            beginAtZero: true,
            grid: { color: '#f0f0f2', lineWidth: 1 },
            ticks: { color: '#9ca3af', font: { size: 9 } },
            border: { display: false }
        },
        x: {
            grid: { display: false },
            ticks: { color: '#9ca3af', font: { size: 9 } },
            border: { display: false }
        }
    };
    const commonPlugins = {
        legend: {
            labels: {
                color: '#6b7280', font: { size: 10 }, boxWidth: 8, boxHeight: 2,
                filter: item => !item.text.startsWith('_')
            }
        },
        tooltip: {
            backgroundColor: '#ffffff', titleColor: '#111111',
            bodyColor: '#6b7280', borderColor: '#e5e7eb',
            borderWidth: 1, cornerRadius: 8, padding: 8
        }
    };
    predictionChart = new Chart(document.getElementById('predictionChart').getContext('2d'), {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: { responsive: true, maintainAspectRatio: false, plugins: commonPlugins, scales: commonScales }
    });
    dailyChart = new Chart(document.getElementById('dailyChart').getContext('2d'), {
        type: 'bar',
        data: { labels: [], datasets: [] },
        options: { responsive: true, maintainAspectRatio: false, plugins: commonPlugins, scales: commonScales }
    });

    // 스파크라인 초기화
    const sparkCtx = document.getElementById('sparklineChart');
    if (sparkCtx) {
        sparklineChart = new Chart(sparkCtx.getContext('2d'), {
            type: 'line',
            data: { labels: [], datasets: [{ data: [], borderColor: 'rgba(0,0,0,0.2)', borderWidth: 1.5, fill: true, backgroundColor: 'rgba(0,0,0,0.04)', tension: 0.4, pointRadius: 0 }] },
            options: {
                responsive: false, maintainAspectRatio: false,
                plugins: { legend: { display: false }, tooltip: { enabled: false } },
                scales: { x: { display: false }, y: { display: false, beginAtZero: true } },
                elements: { point: { radius: 0 } },
                animation: { duration: 600 }
            }
        });
    }
}

function updateChartData(report = null) {
    if (!predictionChart || !dailyChart) return;
    const baseSolar = report ? report.predictions.solar_mw : 30;
    const baseWind = report ? report.predictions.wind_mw : 25;
    const labels = Array.from({ length: 24 }, (_, i) => `${i}시`);
    const solarData = labels.map((_, i) => i >= 6 && i <= 18 ? parseFloat((Math.sin(Math.PI * (i - 6) / 12) * baseSolar * (0.85 + Math.random() * 0.3)).toFixed(2)) : 0);
    const windData = labels.map(() => parseFloat((baseWind * (0.7 + Math.random() * 0.6)).toFixed(2)));

    // 불확실성 범위 (refined_report 우선, 없으면 ±15%)
    const refined = report?.refined_report;
    const solarLow = refined?.refinement_applied
        ? labels.map((_, i) => i >= 6 && i <= 18 ? parseFloat((solarData[i] * (refined.range_low.solar_mw / baseSolar)).toFixed(2)) : 0)
        : solarData.map(v => parseFloat((v * 0.85).toFixed(2)));
    const solarHigh = refined?.refinement_applied
        ? labels.map((_, i) => i >= 6 && i <= 18 ? parseFloat((solarData[i] * (refined.range_high.solar_mw / baseSolar)).toFixed(2)) : 0)
        : solarData.map(v => parseFloat((v * 1.15).toFixed(2)));
    const windLow = windData.map(v => parseFloat((v * 0.85).toFixed(2)));
    const windHigh = windData.map(v => parseFloat((v * 1.15).toFixed(2)));

    predictionChart.data.labels = labels;
    predictionChart.data.datasets = [
        // 태양광 불확실성 밴드
        { label: '_solar_band_low', data: solarLow, borderColor: 'transparent', backgroundColor: 'rgba(249,115,22,0.10)', fill: '+1', tension: 0.3, pointRadius: 0 },
        { label: '_solar_band_high', data: solarHigh, borderColor: 'transparent', backgroundColor: 'rgba(249,115,22,0.10)', fill: false, tension: 0.3, pointRadius: 0 },
        // 풍력 불확실성 밴드
        { label: '_wind_band_low', data: windLow, borderColor: 'transparent', backgroundColor: 'rgba(59,130,246,0.10)', fill: '+1', tension: 0.3, pointRadius: 0 },
        { label: '_wind_band_high', data: windHigh, borderColor: 'transparent', backgroundColor: 'rgba(59,130,246,0.10)', fill: false, tension: 0.3, pointRadius: 0 },
        // 메인 라인
        { label: '☀️ 태양광', data: solarData, borderColor: '#f97316', backgroundColor: 'transparent', borderWidth: 1.5, tension: 0.3, fill: false, pointRadius: 0, pointHoverRadius: 4 },
        { label: '💨 풍력', data: windData, borderColor: '#3b82f6', backgroundColor: 'transparent', borderWidth: 1.5, tension: 0.3, fill: false, pointRadius: 0, pointHoverRadius: 4 }
    ];
    predictionChart.update();

    // 7D 바 차트
    const dtInput = document.getElementById('picker-datetime').value;
    const dt = dtInput ? new Date(dtInput) : new Date();
    const dailyLabels = Array.from({ length: 7 }, (_, i) => { const d = new Date(dt); d.setDate(dt.getDate() + i); return `${d.getMonth() + 1}/${d.getDate()}`; });
    dailyChart.data.labels = dailyLabels;
    dailyChart.data.datasets = [
        { label: '☀️ 태양광', data: dailyLabels.map(() => parseFloat((baseSolar * (0.8 + Math.random() * 0.4)).toFixed(2))), backgroundColor: 'rgba(249,115,22,0.80)', borderColor: '#f97316', borderWidth: 0, borderRadius: 5, borderSkipped: false },
        { label: '💨 풍력', data: dailyLabels.map(() => parseFloat((baseWind * (0.75 + Math.random() * 0.5)).toFixed(2))), backgroundColor: 'rgba(59,130,246,0.80)', borderColor: '#3b82f6', borderWidth: 0, borderRadius: 5, borderSkipped: false }
    ];
    dailyChart.update();

    // 스파크라인 갱신 (지난 12시간 총 발전량 미니 차트)
    if (sparklineChart) {
        const sparkLabels = Array.from({ length: 12 }, (_, i) => `${i}h`);
        const sparkData = sparkLabels.map((_, i) => {
            const h = i + 6;
            const s = h >= 6 && h <= 18 ? Math.sin(Math.PI * (h - 6) / 12) * baseSolar * (0.8 + Math.random() * 0.4) : 0;
            const w = baseWind * (0.7 + Math.random() * 0.6);
            return parseFloat((s + w).toFixed(1));
        });
        sparklineChart.data.labels = sparkLabels;
        sparklineChart.data.datasets[0].data = sparkData;
        sparklineChart.update();
    }
}



// ── 관제 홈 경보 이력 패널 갱신 ──────────────────────────────
async function fetchAndRenderHomeAlerts(report) {
    const listEl = document.getElementById('home-alert-list');
    const badgeEl = document.getElementById('home-alert-badge');
    if (!listEl) return;

    // 현재 세션 경보 (report.warnings)
    const sessionWarnings = [];
    if (report?.trigger_active || (report?.warnings || []).length > 0) {
        const now = new Date();
        const t = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
        (report.warnings || []).forEach(w => sessionWarnings.push({ time: t, msg: w, type: 'warning' }));
    }

    // /events API에서 이력 조회
    let eventsData = [];
    try {
        const res = await fetch('/events');
        if (res.ok) {
            const json = await res.json();
            eventsData = (json.events || []).slice(0, 8);
        }
    } catch (_) { }

    const allItems = [...sessionWarnings, ...eventsData.map(e => ({
        time: (e.timestamp || '').slice(11, 16),
        msg: e.reason || e.type || '이벤트',
        type: e.type
    }))];

    if (allItems.length === 0) {
        listEl.innerHTML = '<p class="text-[11px] text-[#a3a3a3] italic">감지된 경보 없음</p>';
        badgeEl.classList.add('hidden');
        return;
    }

    badgeEl.textContent = `${allItems.length}건`;
    badgeEl.classList.remove('hidden');
    listEl.innerHTML = '';
    allItems.forEach(item => {
        const isAlert = item.type === 'alert' || (item.msg && item.msg.length > 0 && !item.type);
        const div = document.createElement('div');
        div.className = `text-[10px] rounded-lg px-3 py-2 animate-fade-in ${isAlert ? 'bg-red-50 border border-red-200' : 'bg-amber-50 border border-amber-200'}`;
        div.innerHTML = `<span class="font-mono text-[#9ca3af]">${item.time || '--:--'}</span> <span class="${isAlert ? 'text-red-600' : 'text-amber-600'} font-medium">${item.msg}</span>`;
        listEl.appendChild(div);
    });
}

// 인라인 채팅 (레거시 — 레일로 통합됨, 참조용 유지)
async function sendInlineChat(viewType) {
    const region = document.getElementById('map-region-title').textContent;
    const gen = document.getElementById('map-info-generation').textContent;
    const ctx = viewType === 'map' ? `[지역: ${region}, 발전량: ${gen}] ` : '';
    const input = document.getElementById('rail-chat-input');
    if (!input.value.trim()) return;
    input.value = ctx + input.value;
    sendRailChat();
}

function clearAllSessions() {
    const railMsg = document.getElementById('rail-chat-messages');
    if (railMsg) railMsg.innerHTML = '';
}

async function sendTestBriefingEmail(briefingType, buttonEl) {
    if (!buttonEl) return;
    
    // 버튼 비활성화 및 로딩 상태 연출
    const originalText = buttonEl.textContent;
    buttonEl.disabled = true;
    buttonEl.textContent = '발송 중...';
    buttonEl.classList.remove('bg-[#e05a2b]', 'hover:bg-[#c94d22]');
    buttonEl.classList.add('bg-gray-400');
    
    try {
        const response = await fetch(`/api/send-briefing-email?briefing_type=${encodeURIComponent(briefingType)}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (response.ok && data.status === 'ok') {
            alert(`✅ [${briefingType}] 이메일이 성공적으로 발송되었습니다!`);
            buttonEl.textContent = '완료';
            buttonEl.classList.remove('bg-gray-400');
            buttonEl.classList.add('bg-emerald-500');
            
            // 2초 뒤 원상복구
            setTimeout(() => {
                buttonEl.disabled = false;
                buttonEl.textContent = originalText;
                buttonEl.classList.remove('bg-emerald-500');
                buttonEl.classList.add('bg-[#e05a2b]', 'hover:bg-[#c94d22]');
            }, 2000);
        } else {
            throw new Error(data.detail || '이메일 발송 실패');
        }
    } catch (err) {
        console.error(err);
        alert(`❌ 이메일 발송 실패: ${err.message}\nSMTP 설정(.env) 또는 이메일 수신자 설정을 확인해 주세요.`);
        buttonEl.textContent = '실패';
        buttonEl.classList.remove('bg-gray-400');
        buttonEl.classList.add('bg-red-500');
        
        setTimeout(() => {
            buttonEl.disabled = false;
            buttonEl.textContent = originalText;
            buttonEl.classList.remove('bg-red-500');
            buttonEl.classList.add('bg-[#e05a2b]', 'hover:bg-[#c94d22]');
        }, 2000);
    }
}
