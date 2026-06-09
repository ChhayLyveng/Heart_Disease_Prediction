const DEFAULT_ENDPOINT = '/submit-form';
const STATUS_CLASS = 'form-status';
const LOADING_CLASS = 'loading';

function createStatusElement(form) {
    let status = form.querySelector(`.${STATUS_CLASS}`);
    if (!status) {
        status = document.createElement('div');
        status.className = STATUS_CLASS;
        status.setAttribute('aria-live', 'polite');
        status.style.marginTop = '1rem';
        status.style.fontSize = '0.95rem';
        form.appendChild(status);
    }
    return status;
}

function setFormStatus(form, message, type = 'info') {
    const status = createStatusElement(form);
    status.textContent = message;
    status.dataset.state = type;
    status.style.color = type === 'success' ? '#0f766e' : type === 'error' ? '#b91c1c' : '#374151';
}

function setLoadingState(form, isLoading) {
    const submit = form.querySelector('[type="submit"]');
    if (!submit) return;
    submit.disabled = isLoading;
    if (isLoading) {
        submit.dataset.originalText = submit.textContent;
        submit.textContent = 'Sending...';
        form.classList.add(LOADING_CLASS);
    } else {
        submit.textContent = submit.dataset.originalText || submit.textContent;
        form.classList.remove(LOADING_CLASS);
    }
}

export async function handleSubmit(event, options = {}) {
    event.preventDefault();
    const form = event.target;
    if (!(form instanceof HTMLFormElement)) return;

    const endpoint = options.endpoint || form.getAttribute('action') || DEFAULT_ENDPOINT;
    const formData = new FormData(form);

    setLoadingState(form, true);
    setFormStatus(form, 'Sending your message...', 'info');

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorText = await response.text().catch(() => response.statusText || 'Unable to send message.');
            throw new Error(errorText);
        }

        form.reset();
        setFormStatus(form, 'Thank you! Your message has been sent successfully.', 'success');
    } catch (error) {
        setFormStatus(form, `Error: ${error.message || 'Unable to send message.'}`, 'error');
    } finally {
        setLoadingState(form, false);
    }
}

export function initFormHandler(formSelector = '#contactForm', options = {}) {
    const form = document.querySelector(formSelector);
    if (!form) return;

    form.addEventListener('submit', (event) => handleSubmit(event, options));
}

export default handleSubmit;
