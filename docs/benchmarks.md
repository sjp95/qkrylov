# QKrylov Master Scaling Benchmark

<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
.bench-dashboard { margin-top: 1rem; }
.dataset-selector { margin-bottom: 1rem; }
.bench-controls-container { margin-top: 1rem; background: var(--md-code-bg-color); padding: 1.5rem; border-radius: 0.5rem; }
.slider-full-width { width: 100%; margin-bottom: 2rem; }

/* Dual Slider CSS */
.range-slider { position: relative; width: 100%; height: 40px; }
.range-slider input[type="range"] {
    position: absolute; left: 0; top: 50%; transform: translateY(-50%);
    width: 100%; -webkit-appearance: none; background: transparent; pointer-events: none; margin: 0;
}
.range-slider::before {
    content: ''; position: absolute; top: 50%; left: 0; right: 0; height: 4px;
    background: var(--md-default-fg-color--lighter); transform: translateY(-50%); border-radius: 2px;
}
.range-slider input[type="range"]:focus { outline: none; }

/* Min Slider Thumb (Top Triangle) */
.min-slider::-webkit-slider-thumb {
    -webkit-appearance: none; pointer-events: auto; width: 20px; height: 20px;
    background: var(--md-primary-fg-color); clip-path: polygon(0 0, 100% 0, 50% 100%);
    transform: translateY(-14px); cursor: pointer;
}
.min-slider::-moz-range-thumb {
    pointer-events: auto; width: 20px; height: 20px; background: var(--md-primary-fg-color);
    clip-path: polygon(0 0, 100% 0, 50% 100%); transform: translateY(-14px); cursor: pointer; border: none;
}

/* Max Slider Thumb (Bottom Triangle) */
.max-slider::-webkit-slider-thumb {
    -webkit-appearance: none; pointer-events: auto; width: 20px; height: 20px;
    background: var(--md-primary-fg-color); clip-path: polygon(50% 0, 0 100%, 100% 100%);
    transform: translateY(14px); cursor: pointer; opacity: 0.9;
}
.max-slider::-moz-range-thumb {
    pointer-events: auto; width: 20px; height: 20px; background: var(--md-primary-fg-color);
    clip-path: polygon(50% 0, 0 100%, 100% 100%); transform: translateY(14px); cursor: pointer; opacity: 0.9; border: none;
}

.tables-side-by-side { display: flex; flex-wrap: wrap; gap: 2rem; justify-content: space-between; }
.control-group { flex: 1; min-width: 250px; }
.toggle-table { width: 100%; border-collapse: collapse; margin-top: 5px; background: var(--md-default-bg-color); border-radius: 4px; overflow: hidden; }
.toggle-table th, .toggle-table td { border: 1px solid var(--md-default-fg-color--light); padding: 8px; text-align: center; }
.toggle-table th { background: rgba(0,0,0,0.05); }
.toggle-table input[type="checkbox"] { cursor: pointer; transform: scale(1.1); }
#chart { width: 100%; min-height: 500px; }
.md-typeset table:not([class]) th { min-width: 100px; }
</style>

<div class="bench-dashboard">
    <div class="dataset-selector">
        <h3 style="margin-top:0;">Dataset Selection</h3>
        <select id="dataset-select" style="width: 100%; max-width: 400px; padding: 8px; font-size: 1rem; background: var(--md-default-bg-color); color: var(--md-default-fg-color); border: 1px solid var(--md-default-fg-color--light); border-radius: 4px; cursor: pointer;">
            <option value="EarlyStopping">With Early Stopping (Real-World)</option>
            <option value="NoEarlyStopping">No Early Stopping (100 Iters Profiling)</option>
        </select>
        <div id="dataset-message" style="margin-top: 12px; padding: 12px; background: var(--md-code-bg-color); border-left: 4px solid var(--md-primary-fg-color); border-radius: 4px; font-size: 0.95rem; line-height: 1.5;">
            <!-- JS dynamically injects description here -->
        </div>
    </div>

    <div id="chart"></div>

    <div class="bench-controls-container">
        <div class="slider-full-width">
            <h3 style="margin-top:0;">Lattice Size Range: <span id="min-val" style="color: var(--md-primary-fg-color); font-weight: bold;">20</span> to <span id="max-val" style="color: var(--md-primary-fg-color); font-weight: bold;">25</span></h3>
            <div class="range-slider">
                <input type="range" id="min-L" min="20" max="25" value="20" class="min-slider" step="1">
                <input type="range" id="max-L" min="20" max="25" value="25" class="max-slider" step="1">
            </div>
            <div style="display:flex; justify-content:space-between; padding: 0 10px; margin-top: 2px; font-size: 0.85rem; font-family: monospace; color: var(--md-default-fg-color--light);">
                <span>20</span><span>21</span><span>22</span><span>23</span><span>24</span><span>25</span>
            </div>
        </div>

        <div class="tables-side-by-side">
            <div class="control-group">
                <h3 style="margin-top:0;"><label><input type="checkbox" id="toggle-fp32" checked> FP32 (32-bit)</label></h3>
                <table class="toggle-table">
                    <tr>
                        <th></th>
                        <th><label><input type="checkbox" id="toggle-fp32-col-sp" checked> SP</label></th>
                        <th><label><input type="checkbox" id="toggle-fp32-col-tp" checked> TP</label></th>
                    </tr>
                    <tr>
                        <th><label><input type="checkbox" id="toggle-fp32-row-cpu" checked> CPU</label></th>
                        <td><input type="checkbox" class="cb-fp32 cb-fp32-col-sp cb-fp32-row-cpu" id="fp32-cpu-sp" checked></td>
                        <td><input type="checkbox" class="cb-fp32 cb-fp32-col-tp cb-fp32-row-cpu" id="fp32-cpu-tp" checked></td>
                    </tr>
                    <tr>
                        <th><label><input type="checkbox" id="toggle-fp32-row-gpu" checked> GPU</label></th>
                        <td><input type="checkbox" class="cb-fp32 cb-fp32-col-sp cb-fp32-row-gpu" id="fp32-gpu-sp" checked></td>
                        <td><input type="checkbox" class="cb-fp32 cb-fp32-col-tp cb-fp32-row-gpu" id="fp32-gpu-tp" checked></td>
                    </tr>
                </table>
            </div>

            <div class="control-group">
                <h3 style="margin-top:0;"><label><input type="checkbox" id="toggle-fp64" checked> FP64 (64-bit)</label></h3>
                <table class="toggle-table">
                    <tr>
                        <th></th>
                        <th><label><input type="checkbox" id="toggle-fp64-col-sp" checked> SP</label></th>
                        <th><label><input type="checkbox" id="toggle-fp64-col-tp" checked> TP</label></th>
                    </tr>
                    <tr>
                        <th><label><input type="checkbox" id="toggle-fp64-row-cpu" checked> CPU</label></th>
                        <td><input type="checkbox" class="cb-fp64 cb-fp64-col-sp cb-fp64-row-cpu" id="fp64-cpu-sp" checked></td>
                        <td><input type="checkbox" class="cb-fp64 cb-fp64-col-tp cb-fp64-row-cpu" id="fp64-cpu-tp" checked></td>
                    </tr>
                    <tr>
                        <th><label><input type="checkbox" id="toggle-fp64-row-gpu" checked> GPU</label></th>
                        <td><input type="checkbox" class="cb-fp64 cb-fp64-col-sp cb-fp64-row-gpu" id="fp64-gpu-sp" checked></td>
                        <td><input type="checkbox" class="cb-fp64 cb-fp64-col-tp cb-fp64-row-gpu" id="fp64-gpu-tp" checked></td>
                    </tr>
                </table>
            </div>
        </div>
    </div>
</div>

## Raw Data

<div class="md-typeset__scrollwrap">
    <div class="md-typeset__table">
        <table id="raw-data-table">
            <thead>
                <tr>
                    <th>L</th>
                    <th>CPU SP FP32 (s)</th>
                    <th>CPU TP FP32 (s)</th>
                    <th>GPU SP FP32 (s)</th>
                    <th>GPU TP FP32 (s)</th>
                    <th>CPU SP FP64 (s)</th>
                    <th>CPU TP FP64 (s)</th>
                    <th>GPU SP FP64 (s)</th>
                    <th>GPU TP FP64 (s)</th>
                </tr>
            </thead>
            <tbody>
                <!-- JS dynamically populates this based on dropdown -->
            </tbody>
        </table>
    </div>
</div>

<script>
const RAW_DATA = {
    L: [20, 21, 22, 23, 24, 25],
    EarlyStopping: {
        FP32: {
            CPU: { SP: [0.7357, 0.9638, 10.2371, 4.5100, 23.8867, 20.2557],
                   TP: [1.0896, 0.8060, 2.2069, 5.6588, 15.9405, 65.0] },
            GPU: { SP: [0.1329, 0.5267, 2.8411, 1.5502, 13.3861, null],
                   TP: [0.1990, 0.5238, 0.9320, 1.6845, 9.7695, 8.8871] }
        },
        FP64: {
            CPU: { SP: [0.9199, 1.6254, 5.5993, 9.1541, 27.4243, 48.1952],
                   TP: [0.7953, 1.5123, 4.0914, 7.2031, 19.0250, 36.1901] },
            GPU: { SP: [0.3034, 0.7434, 2.0285, 3.6843, null, null],
                   TP: [0.3666, 0.8579, 1.7663, 4.0212, 8.5482, null] }
        }
    },
    NoEarlyStopping: {
        FP32: {
            CPU: { SP: [2.6105, 5.0478, 10.4256, 21.6241, 45.5860, 94.3255],
                   TP: [1.0807, 2.3071, 4.9060, 10.3149, 21.2443, 44.5341] },
            GPU: { SP: [0.4722, 0.9163, 2.8308, 6.7808, 14.0082, null],
                   TP: [0.4413, 0.9888, 2.0611, 4.6986, 9.6783, 21.4963] }
        },
        FP64: {
            CPU: { SP: [3.0912, 6.9023, 17.9426, 39.3649, 83.7170, 170.8979],
                   TP: [1.7518, 3.8765, 8.4719, 17.6806, 36.5048, 76.2911] },
            GPU: { SP: [0.8675, 2.9967, 6.1593, 14.3068, null, null],
                   TP: [0.7677, 2.3396, 3.4758, 10.3469, 16.0353, null] }
        }
    }
};

const colors = {
    'fp32-cpu-sp': '#1f77b4', 'fp32-cpu-tp': '#aec7e8',
    'fp32-gpu-sp': '#2ca02c', 'fp32-gpu-tp': '#98df8a',
    'fp64-cpu-sp': '#d62728', 'fp64-cpu-tp': '#ff9896',
    'fp64-gpu-sp': '#9467bd', 'fp64-gpu-tp': '#c5b0d5'
};

const names = {
    'fp32-cpu-sp': 'FP32 CPU Single-Pass', 'fp32-cpu-tp': 'FP32 CPU Two-Pass',
    'fp32-gpu-sp': 'FP32 GPU Single-Pass', 'fp32-gpu-tp': 'FP32 GPU Two-Pass',
    'fp64-cpu-sp': 'FP64 CPU Single-Pass', 'fp64-cpu-tp': 'FP64 CPU Two-Pass',
    'fp64-gpu-sp': 'FP64 GPU Single-Pass', 'fp64-gpu-tp': 'FP64 GPU Two-Pass'
};

function updateChart() {
    let datasetId = document.getElementById('dataset-select').value;
    let dataset = RAW_DATA[datasetId];

    let minL = parseInt(document.getElementById('min-L').value);
    let maxL = parseInt(document.getElementById('max-L').value);
    
    
    document.getElementById('min-val').innerText = document.getElementById('min-L').value;
    document.getElementById('max-val').innerText = document.getElementById('max-L').value;

    let indices = RAW_DATA.L.map((l, i) => (l >= minL && l <= maxL) ? i : -1).filter(i => i !== -1);
    let x_vals = indices.map(i => `L=${RAW_DATA.L[i]}`);

    let traces = [];

    const configs = [
        { id: 'fp32-cpu-sp', data: dataset.FP32.CPU.SP },
        { id: 'fp32-cpu-tp', data: dataset.FP32.CPU.TP },
        { id: 'fp32-gpu-sp', data: dataset.FP32.GPU.SP },
        { id: 'fp32-gpu-tp', data: dataset.FP32.GPU.TP },
        { id: 'fp64-cpu-sp', data: dataset.FP64.CPU.SP },
        { id: 'fp64-cpu-tp', data: dataset.FP64.CPU.TP },
        { id: 'fp64-gpu-sp', data: dataset.FP64.GPU.SP },
        { id: 'fp64-gpu-tp', data: dataset.FP64.GPU.TP },
    ];

    configs.forEach(cfg => {
        if (document.getElementById(cfg.id).checked) {
            let y_vals = indices.map(i => cfg.data[i] === null ? 0 : cfg.data[i]);
            let text_vals = indices.map(i => {
                let val = cfg.data[i];
                if (val === null) return "OOM";
                if (val === 65.0) return "> 60s";
                return val.toFixed(4) + "s";
            });

            traces.push({
                x: x_vals,
                y: y_vals,
                name: names[cfg.id],
                type: 'bar',
                marker: { color: colors[cfg.id] },
                text: text_vals,
                textposition: 'outside',
                textfont: {size: 11},
                hovertemplate: "<b>%{x}</b><br>%{data.name}: %{text}<extra></extra>"
            });
        }
    });

    
    // Update Message
    let msgEl = document.getElementById('dataset-message');
    if(datasetId === 'EarlyStopping') {
        msgEl.innerHTML = "<strong>Real-World (Early Stopping):</strong> Iterations halt once convergence is reached. This shows why FP64 often beats FP32 in overall wall-clock time: FP64 maintains pristine orthogonality, requiring far fewer iterations to converge compared to FP32's ghost-state pollution.";
    } else {
        msgEl.innerHTML = "<strong>Uniform Profiling (No Early Stopping):</strong> Artificial benchmark forced to exactly 100 iterations. This is strictly for uniform profiling to compare raw throughput and time-per-iteration. In reality, QKrylov always uses early stopping.";
    }

    // Update Table
    let tbody = document.querySelector('#raw-data-table tbody');
    if (tbody) {
        tbody.innerHTML = '';
        RAW_DATA.L.forEach((l, i) => {
            let tr = document.createElement('tr');
            let fmt = (val) => {
                if (val === null) return "<strong>[OOM]</strong>";
                if (val === 65.0) return "> 60s";
                return val.toFixed(4);
            };
            tr.innerHTML = `
                <td><strong>${l}</strong></td>
                <td><code>${fmt(dataset.FP32.CPU.SP[i])}</code></td>
                <td><code>${fmt(dataset.FP32.CPU.TP[i])}</code></td>
                <td><code>${fmt(dataset.FP32.GPU.SP[i])}</code></td>
                <td><code>${fmt(dataset.FP32.GPU.TP[i])}</code></td>
                <td><code>${fmt(dataset.FP64.CPU.SP[i])}</code></td>
                <td><code>${fmt(dataset.FP64.CPU.TP[i])}</code></td>
                <td><code>${fmt(dataset.FP64.GPU.SP[i])}</code></td>
                <td><code>${fmt(dataset.FP64.GPU.TP[i])}</code></td>
            `;
            tbody.appendChild(tr);
        });
    }

    let titleSuffix = datasetId === "EarlyStopping" ? "(With Early Stopping)" : "(No Early Stopping - 100 Iters)";

    let layout = {
        title: { text: `QKrylov Physics Engine Benchmark ${titleSuffix}`, font: {size: 24} },
        barmode: 'group',
        
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: 'var(--md-default-fg-color)' },
        xaxis: { title: 'Lattice Size (L)' },
        yaxis: { title: 'Execution Time (s)' },
        hovermode: 'x unified',
        margin: { l: 60, r: 20, t: 60, b: 60 }
    };

    Plotly.react('chart', traces, layout);
}

// Attach Event Listeners
document.getElementById('dataset-select').addEventListener('change', updateChart);

document.querySelectorAll('input[type="checkbox"]').forEach(cb => {
    if(!cb.id.startsWith('toggle-')) {
        cb.addEventListener('change', updateChart);
    }
});



// Helper to wire up matrix header toggles
function setupMatrixToggle(masterId, targetClass) {
    document.getElementById(masterId).addEventListener('change', (e) => {
        document.querySelectorAll('.' + targetClass).forEach(cb => cb.checked = e.target.checked);
        updateChart();
    });
}

// Master Toggles
document.getElementById('toggle-fp32').addEventListener('change', (e) => {
    document.querySelectorAll('.cb-fp32').forEach(cb => cb.checked = e.target.checked);
    ['toggle-fp32-col-sp', 'toggle-fp32-col-tp', 'toggle-fp32-row-cpu', 'toggle-fp32-row-gpu'].forEach(id => {
        document.getElementById(id).checked = e.target.checked;
    });
    updateChart();
});

document.getElementById('toggle-fp64').addEventListener('change', (e) => {
    document.querySelectorAll('.cb-fp64').forEach(cb => cb.checked = e.target.checked);
    ['toggle-fp64-col-sp', 'toggle-fp64-col-tp', 'toggle-fp64-row-cpu', 'toggle-fp64-row-gpu'].forEach(id => {
        document.getElementById(id).checked = e.target.checked;
    });
    updateChart();
});

// FP32 Row/Col Toggles
setupMatrixToggle('toggle-fp32-col-sp', 'cb-fp32-col-sp');
setupMatrixToggle('toggle-fp32-col-tp', 'cb-fp32-col-tp');
setupMatrixToggle('toggle-fp32-row-cpu', 'cb-fp32-row-cpu');
setupMatrixToggle('toggle-fp32-row-gpu', 'cb-fp32-row-gpu');

// FP64 Row/Col Toggles
setupMatrixToggle('toggle-fp64-col-sp', 'cb-fp64-col-sp');
setupMatrixToggle('toggle-fp64-col-tp', 'cb-fp64-col-tp');
setupMatrixToggle('toggle-fp64-row-cpu', 'cb-fp64-row-cpu');
setupMatrixToggle('toggle-fp64-row-gpu', 'cb-fp64-row-gpu');

// Init
updateChart();


document.getElementById('min-L').addEventListener('input', function() {
    let maxEl = document.getElementById('max-L');
    if (parseInt(this.value) > parseInt(maxEl.value)) { this.value = maxEl.value; }
    updateChart();
});
document.getElementById('max-L').addEventListener('input', function() {
    let minEl = document.getElementById('min-L');
    if (parseInt(this.value) < parseInt(minEl.value)) { this.value = minEl.value; }
    updateChart();
});

</script>
