document.addEventListener('DOMContentLoaded', () => {
    fetchPredictions();
});

async function fetchPredictions() {
    const loader = document.getElementById('loader');
    const grid = document.getElementById('predictions-grid');
    const emptyState = document.getElementById('empty-state');
    
    try {
        const response = await fetch('/api/predictions');
        const data = await response.json();
        
        loader.style.display = 'none';
        
        if (data.success && data.predictions.length > 0) {
            data.predictions.forEach(pred => {
                grid.appendChild(createPredictionCard(pred));
            });
            
            // Trigger animation after brief delay
            setTimeout(() => {
                document.querySelectorAll('.prob-segment').forEach(el => {
                    el.style.width = el.getAttribute('data-width');
                });
            }, 100);
        } else {
            emptyState.style.display = 'block';
        }
    } catch (error) {
        console.error("Error fetching predictions:", error);
        loader.style.display = 'none';
        emptyState.innerHTML = '<p>Error loading predictions. Ensure the backend is running.</p>';
        emptyState.style.display = 'block';
    }
}

function formatDate(isoString) {
    const date = new Date(isoString);
    return date.toLocaleDateString('en-US', { 
        weekday: 'short', 
        month: 'short', 
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function createPredictionCard(pred) {
    const card = document.createElement('div');
    card.className = 'card';
    
    const { goals, corners } = pred;
    
    card.innerHTML = `
        <div class="match-header">
            <div class="match-date">${formatDate(pred.date)}</div>
            <div class="teams">
                <div class="team">
                    <div class="team-name">${pred.home_team}</div>
                    <div class="team-xg">xG: ${goals.home_xg.toFixed(2)}</div>
                </div>
                <div class="vs">VS</div>
                <div class="team">
                    <div class="team-name">${pred.away_team}</div>
                    <div class="team-xg">xG: ${goals.away_xg.toFixed(2)}</div>
                </div>
            </div>
        </div>
        
        <div class="probs-container">
            <div class="probs-labels">
                <span>${pred.home_team} (${goals.win_prob}%)</span>
                <span>Draw (${goals.draw_prob}%)</span>
                <span>${pred.away_team} (${goals.loss_prob}%)</span>
            </div>
            <div class="probs-bar">
                <div class="prob-segment prob-win" style="width: 0%" data-width="${goals.win_prob}%"></div>
                <div class="prob-segment prob-draw" style="width: 0%" data-width="${goals.draw_prob}%"></div>
                <div class="prob-segment prob-loss" style="width: 0%" data-width="${goals.loss_prob}%"></div>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-box">
                <span class="stat-label">Most Likely Score</span>
                <span class="stat-value score">${goals.most_likely_score[0]} - ${goals.most_likely_score[1]}</span>
                <span class="stat-sub">BTTS: ${goals.btts_prob}% | O2.5: ${goals.over_2_5_prob}%</span>
            </div>
            <div class="stat-box">
                <span class="stat-label">Expected Corners</span>
                <span class="stat-value">${corners.total_expected.toFixed(1)}</span>
                <span class="stat-sub">O8.5: ${corners.over_8_5_prob}% | O9.5: ${corners.over_9_5_prob}%</span>
            </div>
        </div>
    `;
    
    return card;
}
