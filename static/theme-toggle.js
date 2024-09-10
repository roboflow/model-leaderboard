// theme-toggle.js
document.addEventListener('DOMContentLoaded', () => {
    const themeToggle = document.getElementById('theme-toggle');
    const themeLink = document.getElementById('theme-link');

    const currentTheme = localStorage.getItem('theme') || 'light';
    if (currentTheme === 'dark') {
        themeLink.href = 'static/dark-theme.css';
        themeToggle.checked = true;
        document.querySelector('.fa-sun').style.display = 'none';
        document.querySelector('.fa-moon').style.display = 'inline';
    } else {
        themeLink.href = 'static/light-theme.css';
        themeToggle.checked = false;
        document.querySelector('.fa-sun').style.display = 'inline';
        document.querySelector('.fa-moon').style.display = 'none';
    }

    themeToggle.addEventListener('change', () => {
        if (themeToggle.checked) {
            themeLink.href = 'static/dark-theme.css';
            localStorage.setItem('theme', 'dark');
            document.querySelector('.fa-sun').style.display = 'none';
            document.querySelector('.fa-moon').style.display = 'inline';
        } else {
            themeLink.href = 'static/light-theme.css';
            localStorage.setItem('theme', 'light');
            document.querySelector('.fa-sun').style.display = 'inline';
            document.querySelector('.fa-moon').style.display = 'none';
        }
    });
});
