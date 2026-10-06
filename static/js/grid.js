const ROW_HEIGHT = 8;
const GAP = 24;

function layoutMasonry() {
    const grid = document.querySelector(".story-grid");
    if (!grid) {
        return;
    }
    const step = ROW_HEIGHT + GAP;
    grid.querySelectorAll(".card").forEach((card) => {
        const height = card.getBoundingClientRect().height;
        const span = Math.ceil((height + GAP) / step);
        card.style.gridRowEnd = `span ${span}`;
    });
}

function initMasonry() {
    const grid = document.querySelector(".story-grid");
    if (!grid) {
        return;
    }
    grid.querySelectorAll(".card img").forEach((img) => {
        if (!img.complete) {
            img.addEventListener("load", layoutMasonry, { once: true });
        }
    });
    layoutMasonry();
}

window.addEventListener("load", layoutMasonry);
window.addEventListener("resize", layoutMasonry);
initMasonry();
