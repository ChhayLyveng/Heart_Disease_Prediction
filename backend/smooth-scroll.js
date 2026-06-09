const DEFAULT_OFFSET = 80;

function isLocalHashLink(link) {
    if (!link || !link.matches('a[href^="#"]')) {
        return false;
    }
    const href = link.getAttribute('href');
    return href !== '#' && !href.startsWith('#!');
}

export function initSmoothScroll(options = {}) {
    const offset = options.offset ?? DEFAULT_OFFSET;

    document.addEventListener('click', (event) => {
        const link = event.target.closest('a[href^="#"]');
        if (!link || !isLocalHashLink(link)) return;

        const targetId = link.getAttribute('href');
        const targetElement = document.querySelector(targetId);
        if (!targetElement) return;

        const url = new URL(link.href, window.location.href);
        if (url.pathname !== window.location.pathname || url.origin !== window.location.origin) {
            return;
        }

        event.preventDefault();

        const rect = targetElement.getBoundingClientRect();
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const targetPosition = rect.top + scrollTop - offset;

        window.scrollTo({ top: targetPosition, behavior: 'smooth' });

        if (window.history && window.history.pushState) {
            window.history.pushState(null, '', targetId);
        }
    });
}

export default initSmoothScroll;
