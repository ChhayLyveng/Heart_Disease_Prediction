const DEFAULT_SELECTOR = '.cards-grid, .work-grid, .contact-grid, .hero-partners, .what-we-do, .our-work, .contact';
const VISIBLE_CLASS = 'visible';
const REVEAL_CLASS = 'reveal';

function prefersReducedMotion() {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

export function initScrollReveal(options = {}) {
    const selector = options.selector || DEFAULT_SELECTOR;
    const threshold = options.threshold ?? 0.16;
    const rootMargin = options.rootMargin ?? '0px 0px -12% 0px';

    const nodes = Array.from(document.querySelectorAll(selector));
    if (!nodes.length) return;

    if (prefersReducedMotion()) {
        nodes.forEach((node) => node.classList.add(VISIBLE_CLASS));
        return;
    }

    nodes.forEach((node) => node.classList.add(REVEAL_CLASS));

    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                entry.target.classList.add(VISIBLE_CLASS);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold, rootMargin });

    nodes.forEach((node) => observer.observe(node));
}

export default initScrollReveal;
