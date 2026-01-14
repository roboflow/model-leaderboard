function formatRunParams(runParams) {
    if (!runParams) {
        return '';
    }
    return Object.entries(runParams)
        .map(([key, value]) => `<li>${key}: ${value}</li>`)
        .join('');
}

function showTooltip(event) {
    const tooltip = event.target.nextElementSibling;
    tooltip.style.display = 'block';

    const updateTooltipPosition = () => {
        FloatingUIDOM.computePosition(event.target, tooltip, {
            placement: 'top',
            middleware: [FloatingUIDOM.offset(10), FloatingUIDOM.flip(), FloatingUIDOM.shift()],
        }).then(({ x, y }) => {
            Object.assign(tooltip.style, {
                left: `${x}px`,
                top: `${y}px`,
            });
        }).catch(err => console.error('Error computing position:', err));
    };

    updateTooltipPosition();
    window.addEventListener('scroll', updateTooltipPosition);

    tooltip._removeScrollListener = () => {
        window.removeEventListener('scroll', updateTooltipPosition);
    };
}

function hideTooltip(event) {
    const tooltip = event.target.nextElementSibling;
    tooltip.style.display = '';
    if (tooltip._removeScrollListener) {
        tooltip._removeScrollListener();
    }
}

function createBadgeCell(metadata) {
    const cell = document.createElement('td');
    cell.style.whiteSpace = 'nowrap';

    // Create a wrapper div for badges to maintain table cell structure
    const badgeWrapper = document.createElement('div');
    badgeWrapper.style.display = 'flex';
    badgeWrapper.style.flexDirection = 'column';
    badgeWrapper.style.gap = '4px';
    badgeWrapper.style.alignItems = 'flex-start';  // Align badges to the left

    if (metadata.github_url) {
        const githubBadge = document.createElement('img');
        githubBadge.src = '/static/github.svg';
        githubBadge.style.height = '20px';
        const githubLink = document.createElement('a');
        githubLink.href = metadata.github_url;
        githubLink.className = 'badge-link';
        githubLink.appendChild(githubBadge);
        badgeWrapper.appendChild(githubLink);
    }

    // Always add arXiv badge - either with link or "no paper" text
    const arxivBadge = document.createElement('img');
    if (metadata.paper_url && metadata.paper_url.includes('arxiv.org')) {
        const arxivId = metadata.paper_url.split('abs/')[1];
        arxivBadge.src = `https://img.shields.io/badge/arXiv-${arxivId}-b31b1b.svg`;
        const paperLink = document.createElement('a');
        paperLink.href = metadata.paper_url;
        paperLink.className = 'badge-link';
        paperLink.appendChild(arxivBadge);
        badgeWrapper.appendChild(paperLink);
    } else {
        arxivBadge.src = `https://img.shields.io/badge/arXiv-no_paper-lightgrey.svg`;
        arxivBadge.style.height = '20px';
        badgeWrapper.appendChild(arxivBadge);
    }

    cell.appendChild(badgeWrapper);
    return cell;
}

function populateTable() {
    const tableBody = document.getElementById('table-body');

    // Results loaded from aggregate_results.json
    results.forEach(result => {
        const row = document.createElement('tr');

        // Create cells
        const modelCell = document.createElement('td');
        modelCell.textContent = result.metadata.model;

        const paramCell = document.createElement('td');
        paramCell.textContent = (result.metadata.param_count / 1e6).toFixed(2);

        const inferenceParamCell = document.createElement('td');
        const gearIcon = document.createElement('i');
        gearIcon.className = 'fas fa-gear';
        gearIcon.style.cursor = 'pointer';

        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip-text';
        tooltip.style.display = 'none';

        const formattedParams = formatRunParams(result.metadata.run_parameters);
        tooltip.innerHTML = `<ul>${formattedParams}</ul>`;

        inferenceParamCell.appendChild(gearIcon);
        inferenceParamCell.appendChild(tooltip);

        [
            ['mouseenter', showTooltip],
            ['mouseleave', hideTooltip],
            ['focus', showTooltip],
            ['blur', hideTooltip],
        ].forEach(([event, listener]) => {
            gearIcon.addEventListener(event, listener);
        });

        const combinedCell = document.createElement('td');
        combinedCell.appendChild(gearIcon);
        combinedCell.appendChild(tooltip);
        combinedCell.appendChild(document.createTextNode(' ' + result.metadata.model));

        const map50_95Cell = document.createElement('td');
        map50_95Cell.textContent = result.map50_95.toFixed(3);

        const map50Cell = document.createElement('td');
        map50Cell.textContent = result.map50.toFixed(3);

        const map75Cell = document.createElement('td');
        map75Cell.textContent = result.map75.toFixed(3);

        const smallMap50_95Cell = document.createElement('td');
        smallMap50_95Cell.textContent = result.small_objects.map50_95.toFixed(3);

        const mediumMap50_95Cell = document.createElement('td');
        mediumMap50_95Cell.textContent = result.medium_objects.map50_95.toFixed(3);

        const largeMap50_95Cell = document.createElement('td');
        largeMap50_95Cell.textContent = result.large_objects.map50_95.toFixed(3);

        const f1_50Cell = document.createElement('td');
        f1_50Cell.textContent = result.f1_50.toFixed(3);

        const f1_75Cell = document.createElement('td');
        f1_75Cell.textContent = result.f1_75.toFixed(3);

        const f1SmallObjects50Cell = document.createElement('td');
        f1SmallObjects50Cell.textContent = result.f1_small_objects.f1_50.toFixed(3);

        const f1SmallObjects75Cell = document.createElement('td');
        f1SmallObjects75Cell.textContent = result.f1_small_objects.f1_75.toFixed(3);

        const f1MediumObjects50Cell = document.createElement('td');
        f1MediumObjects50Cell.textContent = result.f1_medium_objects.f1_50.toFixed(3);

        const f1MediumObjects75Cell = document.createElement('td');
        f1MediumObjects75Cell.textContent = result.f1_medium_objects.f1_75.toFixed(3);

        const f1LargeObjects50Cell = document.createElement('td');
        f1LargeObjects50Cell.textContent = result.f1_large_objects.f1_50.toFixed(3);

        const f1LargeObjects75Cell = document.createElement('td');
        f1LargeObjects75Cell.textContent = result.f1_large_objects.f1_75.toFixed(3);

        const licenseCell = document.createElement('td');
        licenseCell.textContent = result.metadata.license;

        const badgeCell = createBadgeCell(result.metadata || {});

        row.appendChild(combinedCell);
        row.appendChild(paramCell);
        row.appendChild(badgeCell);
        row.appendChild(map50_95Cell);
        row.appendChild(map50Cell);
        row.appendChild(map75Cell);
        row.appendChild(smallMap50_95Cell);
        row.appendChild(mediumMap50_95Cell);
        row.appendChild(largeMap50_95Cell);
        row.appendChild(f1_50Cell);
        row.appendChild(f1_75Cell);
        row.appendChild(f1SmallObjects50Cell);
        row.appendChild(f1SmallObjects75Cell);
        row.appendChild(f1MediumObjects50Cell);
        row.appendChild(f1MediumObjects75Cell);
        row.appendChild(f1LargeObjects50Cell);
        row.appendChild(f1LargeObjects75Cell);
        row.appendChild(licenseCell);

        tableBody.appendChild(row);
    });

    // Fancy table
    new DataTable('#leaderboard', {
        scrollY: '60vh',
        scrollCollapse: true,
        paging: false,
        order: [[3, 'desc']]  // Updated index to 3 since badges column is now at index 2
    });
}

function tooltipHeader() {
    var tooltip = $('.tooltip');
    var delay;

    $('th').hover(function (event) {
        var tooltipText = $(this).data('header-tooltip');
        delay = setTimeout(function () {
            tooltip.text(tooltipText)
                .css({
                    top: event.pageY + 10 + 'px',
                    left: event.pageX + 10 + 'px'
                })
                .fadeIn(200);
        }, 700);
    }, function () {
        clearTimeout(delay);
        tooltip.fadeOut(200);
    });

    $('th').mousemove(function (event) {
        tooltip.css({
            top: event.pageY + 10 + 'px',
            left: event.pageX + 10 + 'px'
        });
    });
};

document.addEventListener('DOMContentLoaded', populateTable);
document.addEventListener('DOMContentLoaded', tooltipHeader);
