// Shared Predictable UI: theme switching, site navigation and team badges.
// Loaded synchronously in <head> so the saved theme applies before first paint.

(function () {
    let theme = null;
    try { theme = localStorage.getItem('predictable.theme'); } catch (e) {}
    if (theme !== 'light' && theme !== 'dark') {
        theme = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    document.documentElement.dataset.theme = theme;
})();

function toggleTheme() {
    const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('predictable.theme', theme); } catch (e) {}
    syncThemeButton();
    window.dispatchEvent(new Event('themechange'));
}

function syncThemeButton() {
    const btn = document.getElementById('themeBtn');
    if (!btn) return;
    const dark = document.documentElement.dataset.theme === 'dark';
    btn.setAttribute('aria-label', dark ? 'Switch to white background' : 'Switch to dark background');
}

// Read a theme colour token, e.g. themeColor('--ink-2'), for canvas drawing
function themeColor(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function esc(v) {
    return String(v ?? '').replace(/[&<>"]/g, ch => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;' }[ch]));
}

// === Navigation ===
const NAV = [
    { href: '/epl', label: 'Predictions', short: 'Predictions',
      icon: '<circle cx="12" cy="12" r="9"/><path d="M12 7l4 3-1.5 4.5h-5L8 10z"/><path d="M12 3v4M21 10.5l-5-.5M17.5 19l-3-4.5M6.5 19l3-4.5M3 10.5l5-.5"/>' },
    { href: '/stats', label: 'Stats', short: 'Stats',
      icon: '<path d="M12 3a9 9 0 109 9h-9z"/><path d="M15 3.5A9 9 0 0120.5 9H15z"/>' },
    { href: '/ucl', label: 'Champions League', short: 'UCL',
      icon: '<path d="M12 3l2.2 4.5 4.8.7-3.5 3.4.8 4.9L12 14.2l-4.3 2.3.8-4.9L5 8.2l4.8-.7z"/><path d="M8 21h8M12 17v4"/>' },
    { href: '/calculator', label: 'Acca Calculator', short: 'Calculator',
      icon: '<rect x="5" y="3" width="14" height="18" rx="2.5"/><path d="M8.5 7h7M8.5 11h1M12 11h1M8.5 14.5h1M12 14.5h1M8.5 18h1M12 18h1M15.5 11v7"/>' },
    { href: '/backtest', label: 'Backtest Results', short: 'Backtest',
      icon: '<path d="M4 20V4M4 20h16"/><path d="M7 15l4-4 3 3 5-6"/>' },
];

const THEME_BUTTON = `
    <button class="theme-btn" id="themeBtn" onclick="toggleTheme()">
        <svg class="moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1111.2 3a7 7 0 009.8 9.8z"/></svg>
        <svg class="sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
    </button>`;

// Fills <div class="hero-actions" data-nav="top"> and <nav class="bottom-nav" data-nav="bottom">
function renderNav() {
    const path = location.pathname === '/' ? '/epl' : location.pathname.replace(/\/$/, '');
    const isActive = item => item.href === path;
    const top = document.querySelector('[data-nav="top"]');
    if (top) {
        top.innerHTML = `<nav class="top-links" aria-label="Main">${NAV.map(item =>
            `<a href="${item.href}"${isActive(item) ? ' class="active" aria-current="page"' : ''}>${item.label}</a>`).join('')}</nav>${THEME_BUTTON}`;
    }
    const bottom = document.querySelector('[data-nav="bottom"]');
    if (bottom) {
        bottom.innerHTML = NAV.map(item => `
            <a href="${item.href}"${isActive(item) ? ' class="active" aria-current="page"' : ''}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${item.icon}</svg>
                ${item.short}
            </a>`).join('');
    }
    syncThemeButton();
}
document.addEventListener('DOMContentLoaded', renderNav);

// === Team badges ===
// display name, 3-letter code, badge colours
const TEAMS = {
    'Arsenal': ['Arsenal', 'ARS', '#EF0107', '#ffffff'],
    'Aston Villa': ['Aston Villa', 'AVL', '#670E36', '#95BFE5'],
    'Bournemouth': ['Bournemouth', 'BOU', '#DA291C', '#111111'],
    'Brentford': ['Brentford', 'BRE', '#E30613', '#ffffff'],
    'Brighton & Hove Albion': ['Brighton', 'BHA', '#0057B8', '#ffffff'],
    'Burnley': ['Burnley', 'BUR', '#6C1D45', '#99D6EA'],
    'Chelsea': ['Chelsea', 'CHE', '#034694', '#ffffff'],
    'Coventry City': ['Coventry', 'COV', '#59CBE8', '#ffffff'],
    'Crystal Palace': ['Crystal Palace', 'CRY', '#1B458F', '#ffffff'],
    'Everton': ['Everton', 'EVE', '#003399', '#ffffff'],
    'Fulham': ['Fulham', 'FUL', '#ffffff', '#111111'],
    'Hull City': ['Hull', 'HUL', '#F5A12D', '#111111'],
    'Ipswich Town': ['Ipswich', 'IPS', '#3A64A3', '#ffffff'],
    'Leeds United': ['Leeds', 'LEE', '#FFCD00', '#1D428A'],
    'Leicester City': ['Leicester', 'LEI', '#003090', '#FDBE11'],
    'Liverpool': ['Liverpool', 'LIV', '#C8102E', '#ffffff'],
    'Luton Town': ['Luton', 'LUT', '#F78F1E', '#002D62'],
    'Manchester City': ['Man City', 'MCI', '#6CABDD', '#1C2C5B'],
    'Manchester United': ['Man Utd', 'MUN', '#DA291C', '#FBE122'],
    'Newcastle United': ['Newcastle', 'NEW', '#241F20', '#ffffff'],
    'Nottingham Forest': ["Nott'm Forest", 'NFO', '#DD0000', '#ffffff'],
    'Sheffield United': ['Sheffield Utd', 'SHU', '#EE2737', '#ffffff'],
    'Southampton': ['Southampton', 'SOU', '#D71920', '#ffffff'],
    'Sunderland AFC': ['Sunderland', 'SUN', '#EB172B', '#ffffff'],
    'Tottenham': ['Spurs', 'TOT', '#132257', '#ffffff'],
    'West Ham United': ['West Ham', 'WHU', '#7A263A', '#1BB1E7'],
    'Wolverhampton Wanderers': ['Wolves', 'WOL', '#FDB913', '#231F20'],
};
// Shorter spellings used by older seasons' data
const TEAM_ALIASES = {
    'Brighton': 'Brighton & Hove Albion', 'Ipswich': 'Ipswich Town', 'Leicester': 'Leicester City',
    'Newcastle': 'Newcastle United', 'West Ham': 'West Ham United', 'Wolves': 'Wolverhampton Wanderers',
    'Sunderland': 'Sunderland AFC', 'Leeds': 'Leeds United', 'Hull': 'Hull City', 'Coventry': 'Coventry City',
    'Luton': 'Luton Town', 'Sheffield Utd': 'Sheffield United',
};

function team(name) {
    const t = TEAMS[name] || TEAMS[TEAM_ALIASES[name]] || [name, name.slice(0, 3).toUpperCase(), '#963cff', '#ffffff'];
    return { short: t[0], code: t[1], c1: t[2], c2: t[3] };
}

// Club crest (64px WebP, ~2KB, cached), falling back to a coloured badge with the club code
function crest(name) {
    const t = team(name);
    return `<span class="crest has-img" style="--c1:${t.c1};--c2:${t.c2}" aria-hidden="true">`
        + `<img src="/static/img/crests/${esc(t.code)}.webp" alt="" width="28" height="28" loading="lazy" decoding="async" onerror="crestFallback(this)"></span>`;
}

function crestFallback(img) {
    const badge = img.parentNode;
    badge.classList.remove('has-img');
    badge.textContent = badge.querySelector('img').src.match(/(\w+)\.webp$/)[1];
}
