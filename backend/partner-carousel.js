const AUTO_ADVANCE_DELAY = 3500;
const SCROLL_AMOUNT = 220;

function createButton(label, className) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = className;
    button.textContent = label;
    button.style.border = 'none';
    button.style.background = 'rgba(255,255,255,0.9)';
    button.style.color = '#0b2d45';
    button.style.fontWeight = '700';
    button.style.cursor = 'pointer';
    button.style.padding = '0.75rem 1rem';
    button.style.borderRadius = '999px';
    button.style.boxShadow = '0 8px 18px rgba(0,0,0,0.08)';
    return button;
}

export function initPartnerCarousel(options = {}) {
    const wrapper = document.querySelector('.hero-partners');
    if (!wrapper) return;

    const cards = Array.from(wrapper.children);
    if (cards.length < 5) return;

    wrapper.style.display = 'flex';
    wrapper.style.flexWrap = 'nowrap';
    wrapper.style.overflowX = 'auto';
    wrapper.style.scrollBehavior = 'smooth';
    wrapper.style.padding = '1rem 0';
    wrapper.style.gap = '1rem';
    wrapper.style.scrollSnapType = 'x mandatory';

    cards.forEach((card) => {
        card.style.flex = '0 0 auto';
        card.style.scrollSnapAlign = 'center';
    });

    const container = document.createElement('div');
    container.className = 'partner-carousel-controls';
    container.style.position = 'relative';
    container.style.display = 'flex';
    container.style.justifyContent = 'space-between';
    container.style.alignItems = 'center';
    container.style.marginTop = '1rem';
    wrapper.parentNode.insertBefore(container, wrapper.nextSibling);

    const prevButton = createButton('‹', 'partner-carousel-prev');
    const nextButton = createButton('›', 'partner-carousel-next');

    prevButton.addEventListener('click', () => {
        wrapper.scrollBy({ left: -SCROLL_AMOUNT, behavior: 'smooth' });
    });
    nextButton.addEventListener('click', () => {
        wrapper.scrollBy({ left: SCROLL_AMOUNT, behavior: 'smooth' });
    });

    container.appendChild(prevButton);
    container.appendChild(nextButton);

    let autoScrollTimer = null;
    const startAutoScroll = () => {
        if (autoScrollTimer) return;
        autoScrollTimer = window.setInterval(() => {
            if (wrapper.scrollLeft + wrapper.clientWidth >= wrapper.scrollWidth - 10) {
                wrapper.scrollTo({ left: 0, behavior: 'smooth' });
                return;
            }
            wrapper.scrollBy({ left: SCROLL_AMOUNT, behavior: 'smooth' });
        }, options.delay ?? AUTO_ADVANCE_DELAY);
    };

    const stopAutoScroll = () => {
        if (!autoScrollTimer) return;
        window.clearInterval(autoScrollTimer);
        autoScrollTimer = null;
    };

    wrapper.addEventListener('mouseenter', stopAutoScroll);
    wrapper.addEventListener('mouseleave', startAutoScroll);
    prevButton.addEventListener('mouseenter', stopAutoScroll);
    nextButton.addEventListener('mouseenter', stopAutoScroll);
    prevButton.addEventListener('mouseleave', startAutoScroll);
    nextButton.addEventListener('mouseleave', startAutoScroll);

    startAutoScroll();
}

export default initPartnerCarousel;
