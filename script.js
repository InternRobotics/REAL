const root = document.documentElement;
const header = document.querySelector('.site-header');
const themeToggle = document.querySelector('#theme-toggle');
const menuButton = document.querySelector('#menu-button');
const mobileNav = document.querySelector('#mobile-nav');
const copyButton = document.querySelector('#copy-citation');
const backToTopButton = document.querySelector('#back-to-top');

function preferredTheme() {
    const savedTheme = localStorage.getItem('project-page-theme');
    if (savedTheme === 'light' || savedTheme === 'dark') return savedTheme;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function applyTheme(theme) {
    root.dataset.theme = theme;
    themeToggle.setAttribute('aria-pressed', String(theme === 'dark'));
    themeToggle.setAttribute('aria-label', theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
}

applyTheme(preferredTheme());

themeToggle.addEventListener('click', () => {
    const nextTheme = root.dataset.theme === 'dark' ? 'light' : 'dark';
    applyTheme(nextTheme);
    localStorage.setItem('project-page-theme', nextTheme);
});

function closeMenu() {
    menuButton.setAttribute('aria-expanded', 'false');
    menuButton.setAttribute('aria-label', 'Open navigation');
    mobileNav.hidden = true;
    document.body.classList.remove('menu-open');
}

menuButton.addEventListener('click', () => {
    const shouldOpen = menuButton.getAttribute('aria-expanded') !== 'true';
    menuButton.setAttribute('aria-expanded', String(shouldOpen));
    menuButton.setAttribute('aria-label', shouldOpen ? 'Close navigation' : 'Open navigation');
    mobileNav.hidden = !shouldOpen;
    document.body.classList.toggle('menu-open', shouldOpen);
});

mobileNav.querySelectorAll('a').forEach((link) => link.addEventListener('click', closeMenu));

window.addEventListener('resize', () => {
    if (window.innerWidth > 720) closeMenu();
});

async function copyCitation() {
    const citation = document.querySelector('#bibtex').textContent.trim();
    const label = copyButton.querySelector('.copy-label');
    try {
        await navigator.clipboard.writeText(citation);
        label.textContent = 'Copied';
    } catch (error) {
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(document.querySelector('#bibtex'));
        selection.removeAllRanges();
        selection.addRange(range);
        label.textContent = 'Selected';
    }
    window.setTimeout(() => { label.textContent = 'Copy'; }, 1800);
}

copyButton.addEventListener('click', copyCitation);

function updateScrollControls() {
    header.classList.toggle('scrolled', window.scrollY > 12);
    backToTopButton.classList.toggle('visible', window.scrollY > 600);
}

window.addEventListener('scroll', updateScrollControls, { passive: true });
updateScrollControls();

backToTopButton.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));

const revealObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
    });
}, { threshold: 0.12 });

document.querySelectorAll('.reveal').forEach((element) => revealObserver.observe(element));

const trajectoryTabs = document.querySelectorAll('.trajectory-tab');
const trajectoryPanels = document.querySelectorAll('.trajectory-panel');

trajectoryTabs.forEach((tab) => {
    tab.addEventListener('click', () => {
        const target = tab.dataset.trajectory;
        trajectoryTabs.forEach((item) => {
            const active = item === tab;
            item.classList.toggle('is-active', active);
            item.setAttribute('aria-selected', String(active));
        });
        trajectoryPanels.forEach((panel) => {
            const active = panel.dataset.panel === target;
            panel.hidden = !active;
            panel.classList.toggle('is-active', active);
            panel.querySelectorAll('video').forEach((video) => {
                if (!active) video.pause();
            });
        });
    });
});

const benchmarkContent = {
    fdp: {
        kicker: 'FDP / ACTIVE EXPLORATION',
        title: 'Object and furniture distractors make visual search necessary.',
        copy: 'The agent must first identify the right source and destination among similarly named or visually similar furniture, then complete a cross-receptacle rearrangement.',
        signalTitle: 'Evidence must precede the pick-and-place action.',
        steps: [
            ['Observe', 'Separate similar furniture from the RGB view.'],
            ['Act', 'Explore and ground the source and destination.'],
            ['Verify', 'Complete the correct cross-receptacle placement.']
        ]
    },
    fodp: {
        kicker: 'FODP / OPEN-VOCABULARY CLUTTER',
        title: 'Search must resolve distractors at both object and furniture levels.',
        copy: 'FODP tests whether the policy can use visual evidence to distinguish open-vocabulary object clutter while also locating the correct receptacles.',
        signalTitle: 'The target must be grounded amid two layers of clutter.',
        steps: [
            ['Observe', 'Inspect furniture and objects without an oracle inventory.'],
            ['Disambiguate', 'Use visual grounding to reject similar candidates.'],
            ['Verify', 'Manipulate the object at the intended receptacle.']
        ]
    },
    fdo: {
        kicker: 'FDO / ARTICULATED MANIPULATION',
        title: 'Opening and closing changes what the agent must remember.',
        copy: 'Dynamic articulated state changes require the policy to keep perception, action history, and manipulation tools synchronized over a longer horizon.',
        signalTitle: 'The policy must maintain state across a changing scene.',
        steps: [
            ['Observe', 'Read the initial articulated state from RGB.'],
            ['Manipulate', 'Open or close the target with a deployable tool.'],
            ['Remember', 'Use the updated state for the remaining plan.']
        ]
    },
    sul: {
        kicker: 'SUL / INTERACTIVE DISAMBIGUATION',
        title: 'The correct action can be to ask for information.',
        copy: 'SUL deliberately withholds enough detail to make immediate action unsafe. The policy must recognize ambiguity and request clarification through the simulated user.',
        signalTitle: 'Dialogue is an action when the observation is insufficient.',
        steps: [
            ['Detect', 'Recognize that the instruction is under-specified.'],
            ['Ask', 'Query the simulated user before manipulating.'],
            ['Ground', 'Use the response to select the intended target.']
        ]
    }
};

const benchmarkTabs = document.querySelectorAll('.benchmark-card');
const benchmarkKicker = document.querySelector('#benchmark-kicker');
const benchmarkTitle = document.querySelector('#benchmark-title');
const benchmarkCopy = document.querySelector('#benchmark-copy');
const benchmarkSignalTitle = document.querySelector('#benchmark-signal-title');
const benchmarkStepTitles = [1, 2, 3].map((index) => document.querySelector(`#benchmark-step-${index}-title`));
const benchmarkStepCopies = [1, 2, 3].map((index) => document.querySelector(`#benchmark-step-${index}-copy`));

benchmarkTabs.forEach((tab) => {
    tab.addEventListener('click', () => {
        const selection = benchmarkContent[tab.dataset.benchmark];
        benchmarkTabs.forEach((item) => {
            const active = item === tab;
            item.classList.toggle('is-active', active);
            item.setAttribute('aria-selected', String(active));
        });
        benchmarkKicker.textContent = selection.kicker;
        benchmarkTitle.textContent = selection.title;
        benchmarkCopy.textContent = selection.copy;
        benchmarkSignalTitle.textContent = selection.signalTitle;
        selection.steps.forEach(([title, copy], index) => {
            benchmarkStepTitles[index].textContent = title;
            benchmarkStepCopies[index].textContent = copy;
        });
    });
});

const resultContent = {
    fdp: {
        kicker: 'FDP / TOOL ALIGNMENT',
        title: 'SFT establishes a usable tool policy.',
        copy: 'One SFT epoch lifts the base model from 0.0% to 45.8%; the second epoch reaches 65.3%.',
        values: [45.8, 65.3, 58.3]
    },
    fodp: {
        kicker: 'FODP / EXPLORATION RECOVERY',
        title: 'Closed-loop RL recovers open-vocabulary exploration.',
        copy: 'After a second SFT epoch drops to 28.6%, online RL restores FODP success to 33.9%.',
        values: [30.4, 28.6, 33.9]
    },
    fdo: {
        kicker: 'FDO / DYNAMIC STATE TRACKING',
        title: 'Articulated manipulation remains a visual bottleneck.',
        copy: 'Two-epoch SFT reaches 43.8% on open/close tasks; the final policy keeps a competitive 41.7%.',
        values: [35.4, 43.8, 41.7]
    },
    sul: {
        kicker: 'SUL / PROACTIVE INTERACTION',
        title: 'RL improves when the policy pauses to ask.',
        copy: 'The interaction-heavy SUL split rises from 50.8% after two SFT epochs to 56.9% with online RL.',
        values: [36.9, 50.8, 56.9]
    }
};

const resultTabs = document.querySelectorAll('.result-tab');
const resultKicker = document.querySelector('#result-chart-kicker');
const resultTitle = document.querySelector('#result-chart-title');
const resultCopy = document.querySelector('#result-chart-copy');
const barElements = [document.querySelector('#bar-sft1'), document.querySelector('#bar-sft2'), document.querySelector('#bar-rl')];
const valueElements = [document.querySelector('#value-sft1'), document.querySelector('#value-sft2'), document.querySelector('#value-rl')];

resultTabs.forEach((tab) => {
    tab.addEventListener('click', () => {
        const selection = resultContent[tab.dataset.result];
        resultTabs.forEach((item) => {
            const active = item === tab;
            item.classList.toggle('is-active', active);
            item.setAttribute('aria-selected', String(active));
        });
        resultKicker.textContent = selection.kicker;
        resultTitle.textContent = selection.title;
        resultCopy.textContent = selection.copy;
        selection.values.forEach((value, index) => {
            barElements[index].style.setProperty('--bar-value', value);
            valueElements[index].textContent = `${value.toFixed(1)}%`;
        });
    });
});
