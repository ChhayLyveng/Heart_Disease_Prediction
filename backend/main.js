import initSmoothScroll from './smooth-scroll.js';
import { initFormHandler } from './form-handler.js';
import initScrollReveal from './scroll-reveal.js';
import initNavbar from './navbar.js';
import initPartnerCarousel from './partner-carousel.js';

window.addEventListener('DOMContentLoaded', () => {
    initSmoothScroll({ offset: 96 });
    initNavbar({ offset: 96 });
    initFormHandler('#contactForm', { endpoint: '/submit-form' });
    initScrollReveal();
    initPartnerCarousel({ delay: 3500 });
});
