/**
 * HKJC 賽馬預測系統 - Frontend JavaScript
 * Live odds polling, countdown timers, and UI utilities.
 */

// ── Live Clock ─────────────────────────────────────────────────────────────
(function startClock() {
  const el = document.getElementById("live-clock");
  if (!el) return;
  function updateClock() {
    const now = new Date();
    const hkt = new Date(now.toLocaleString("en-US", { timeZone: "Asia/Hong_Kong" }));
    const h = String(hkt.getHours()).padStart(2, "0");
    const m = String(hkt.getMinutes()).padStart(2, "0");
    const s = String(hkt.getSeconds()).padStart(2, "0");
    el.textContent = `HKT ${h}:${m}:${s}`;
  }
  updateClock();
  setInterval(updateClock, 1000);
})();


// ── Countdown Timers ────────────────────────────────────────────────────────
/**
 * Start a countdown timer for a race.
 * @param {HTMLElement} el - Element to update
 * @param {string} timeStr - Race time in "HH:MM" format (HKT)
 */
function startCountdown(el, timeStr) {
  if (!el || !timeStr) return;

  function update() {
    const now = new Date();
    const hkt = new Date(now.toLocaleString("en-US", { timeZone: "Asia/Hong_Kong" }));
    const [hours, minutes] = timeStr.split(":").map(Number);

    const raceTime = new Date(hkt);
    raceTime.setHours(hours, minutes, 0, 0);

    const diffMs = raceTime - hkt;
    if (diffMs < 0) {
      el.textContent = "已開賽";
      el.style.color = "#8b949e";
      return;
    }

    const totalMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(totalMins / 60);
    const diffMins = totalMins % 60;

    if (diffHours > 2) {
      el.textContent = `${diffHours}小時後`;
      el.style.color = "#8b949e";
    } else if (totalMins > 30) {
      el.textContent = `${totalMins}分鐘後`;
      el.style.color = "#f5a623";
    } else if (totalMins > 10) {
      el.textContent = `🟡 ${totalMins}分鐘後`;
      el.style.color = "#f5a623";
    } else {
      el.textContent = `🔴 ${totalMins}分鐘後`;
      el.style.color = "#da3633";
      el.style.fontWeight = "700";
    }
  }

  update();
  setInterval(update, 30000);
}


// ── Live Odds Polling ───────────────────────────────────────────────────────
/**
 * Poll the odds API and update displayed values with flash animation.
 * @param {number} raceNumber - Race number to poll
 * @param {number} intervalMs - Poll interval in milliseconds (default 60000)
 */
function startOddsPolling(raceNumber, intervalMs = 60000) {
  let previousOdds = {};
  const lastUpdateEl = document.getElementById("last-update-time");

  async function fetchOdds() {
    try {
      const resp = await fetch(`/api/odds/${raceNumber}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();

      const newOdds = data.odds || {};

      // Update all odds elements
      document.querySelectorAll(".odds-val[data-horse]").forEach(el => {
        const horseNum = el.dataset.horse;
        const oddsType = el.dataset.type;  // "win" or "place"
        const newOddsEntry = newOdds[horseNum];
        if (!newOddsEntry) return;

        const newVal = oddsType === "win"
          ? newOddsEntry.win_odds
          : newOddsEntry.place_odds;

        if (newVal === undefined || newVal === null) return;

        const prevVal = previousOdds[`${horseNum}_${oddsType}`];
        const formatted = parseFloat(newVal).toFixed(1);

        if (prevVal && Math.abs(prevVal - newVal) > 0.01) {
          // Flash animation for changed odds
          el.classList.remove("flashing");
          void el.offsetWidth; // reflow
          el.classList.add("flashing");
          el.style.color = newVal < prevVal ? "#2ea043" : "#da3633";
          setTimeout(() => { el.style.color = ""; }, 2000);
        }

        el.textContent = formatted;
        previousOdds[`${horseNum}_${oddsType}`] = newVal;
      });

      // Update timestamp
      if (lastUpdateEl) {
        const now = new Date(data.updated_at || Date.now());
        lastUpdateEl.textContent = now.toLocaleTimeString("zh-HK", {
          timeZone: "Asia/Hong_Kong",
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        });
      }

    } catch (err) {
      console.warn("Odds fetch failed:", err);
    }
  }

  // Initial fetch + set interval
  fetchOdds();
  setInterval(fetchOdds, intervalMs);
}


// ── Data Refresh ────────────────────────────────────────────────────────────
/**
 * Force a server-side data refresh and reload the page.
 */
async function refreshData() {
  const btn = event?.currentTarget;
  if (btn) {
    btn.innerHTML = '<i class="bi bi-arrow-repeat spin me-1"></i>刷新中...';
    btn.style.pointerEvents = "none";
  }
  try {
    await fetch("/api/refresh");
    location.reload();
  } catch {
    location.reload();
  }
}


// ── Confidence Bar Colour Gradient ──────────────────────────────────────────
document.querySelectorAll(".confidence-bar").forEach(bar => {
  const pct = parseInt(bar.dataset.confidence || "0", 10);
  if (pct >= 70) bar.style.background = "linear-gradient(90deg, #2ea043, #5cb85c)";
  else if (pct >= 50) bar.style.background = "linear-gradient(90deg, #f5a623, #f8c060)";
  else bar.style.background = "linear-gradient(90deg, #da3633, #f06060)";
});


// ── Tooltip initialisation (Bootstrap) ─────────────────────────────────────
document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => {
  new bootstrap.Tooltip(el, { trigger: "hover" });
});


// ── Spin animation ──────────────────────────────────────────────────────────
const style = document.createElement("style");
style.textContent = `
  @keyframes spin {
    0%   { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
  .spin { display: inline-block; animation: spin 0.8s linear infinite; }
`;
document.head.appendChild(style);


// ── Race Card keyboard navigation ───────────────────────────────────────────
document.querySelectorAll(".race-card").forEach(card => {
  card.setAttribute("tabindex", "0");
  card.setAttribute("role", "button");
  card.addEventListener("keydown", e => {
    if (e.key === "Enter" || e.key === " ") card.click();
  });
});
