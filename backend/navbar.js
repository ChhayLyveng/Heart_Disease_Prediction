const SECTION_LINK_SELECTOR = 'nav a[href^="#"]';
const NAV_OPEN_CLASS = 'open';
const NAV_HIDDEN_CLASS = 'hidden';
const ACTIVE_CLASS = 'active';

function getOffsetTop(element, offset = 80) {
    const rect = element.getBoundingClientRect();
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    return rect.top + scrollTop - offset;
}

function setActiveNavLink(navLinks, activeId) {
    navLinks.forEach((link) => {
        const href = link.getAttribute('href');
        if (href === activeId) {
            link.classList.add(ACTIVE_CLASS);
        } else {
            link.classList.remove(ACTIVE_CLASS);
        }
    });
}

function updateActiveSection(navLinks, sections, offset) {
    const scrollPosition = window.pageYOffset || document.documentElement.scrollTop;
    let activeId = sections[0]?.id ? `#${sections[0].id}` : null;

    sections.forEach((section) => {
        if (section.offsetTop - offset <= scrollPosition + 1) {
            activeId = `#${section.id}`;
        }
    });

    setActiveNavLink(navLinks, activeId);
}

function shouldHideNavbar(currentScroll, lastScroll, tolerance = 10) {
    return currentScroll > lastScroll + tolerance;
}

export function initNavbar(options = {}) {
    const navbar = document.getElementById('navbar');
    const navLinksContainer = document.getElementById('navLinks');
    const navToggle = document.getElementById('navToggle');
    const navLinks = Array.from(document.querySelectorAll(SECTION_LINK_SELECTOR));
    const offset = options.offset ?? 96;
    const sections = navLinks
        .map((link) => document.querySelector(link.getAttribute('href')))
        .filter(Boolean);

    if (!navbar || !navLinksContainer || !navToggle || !navLinks.length) return;

    navLinks.forEach((link) => {
        link.addEventListener('click', () => {
            navLinksContainer.classList.remove(NAV_OPEN_CLASS);
        });
    });

    navToggle.addEventListener('click', () => {
        navLinksContainer.classList.toggle(NAV_OPEN_CLASS);
    });

    let lastScrollY = window.pageYOffset || document.documentElement.scrollTop;

    window.addEventListener('scroll', () => {
        const currentScrollY = window.pageYOffset || document.documentElement.scrollTop;

        updateActiveSection(navLinks, sections, offset);

        if (currentScrollY > 120) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }

        if (shouldHideNavbar(currentScrollY, lastScrollY)) {
            navbar.classList.add(NAV_HIDDEN_CLASS);
        } else if (currentScrollY < lastScrollY || currentScrollY < 150) {
            navbar.classList.remove(NAV_HIDDEN_CLASS);
        }

        lastScrollY = currentScrollY;
    });

    updateActiveSection(navLinks, sections, offset);
}

export default initNavbar;
