// Simple scroll animation for About and Tech sections
document.addEventListener('DOMContentLoaded', () => {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate');
            }
        });
    }, { threshold: 0.5 });

    document.querySelectorAll('.about, .tech').forEach(section => {
        observer.observe(section);
    });
});

// Add 'animate' class styles in CSS
document.styleSheets[0].insertRule(`
    .animate {
        animation: fadeIn 1s ease forwards;
    }
`, 0);