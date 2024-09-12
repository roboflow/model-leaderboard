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
    FloatingUI.computePosition(event.target, tooltip, {
        placement: 'top',
        middleware: [FloatingUI.offset(10), FloatingUI.flip(), FloatingUI.shift()],
    }).then(({ x, y }) => {
        Object.assign(tooltip.style, {
            left: `${x}px`,
            top: `${y}px`,
        });
    });
}

function hideTooltip(event) {
    const tooltip = event.target.nextElementSibling;
    tooltip.style.display = 'none';
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
        gearIcon.className = 'fas fa-sliders';
        gearIcon.style.cursor = 'pointer';

        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip-text';
        tooltip.style.display = 'none';
        tooltip.style.position = 'absolute';
        tooltip.style.backgroundColor = 'black';
        tooltip.style.color = 'white';
        tooltip.style.padding = '5px';
        tooltip.style.borderRadius = '6px';
        tooltip.style.zIndex = '1000';

        const formattedParams = formatRunParams(result.metadata.run_parameters);
        tooltip.innerHTML = `<ul>${formattedParams}</ul>`;

        inferenceParamCell.appendChild(gearIcon);
        inferenceParamCell.appendChild(tooltip);

        gearIcon.addEventListener('mouseenter', showTooltip);
        gearIcon.addEventListener('mouseleave', hideTooltip);

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

        const licenseCell = document.createElement('td');
        licenseCell.textContent = result.metadata.license;

        // Append cells to row
        row.appendChild(combinedCell);
        row.appendChild(paramCell);
        row.appendChild(map50_95Cell);
        row.appendChild(map50Cell);
        row.appendChild(map75Cell);
        row.appendChild(smallMap50_95Cell);
        row.appendChild(mediumMap50_95Cell);
        row.appendChild(largeMap50_95Cell);
        row.appendChild(licenseCell);

        // Append row to table body
        tableBody.appendChild(row);
    });

    // Fancy table
    new DataTable('#leaderboard', {
        scrollY: '60vh',   // Set table height to 50% of the viewport height
        scrollCollapse: true,  // Collapse the table if it has fewer rows
        paging: false,  // Disable pagination
        order: [[2, 'desc']]  // Sort by MAP@50-95
    });
}

document.addEventListener('DOMContentLoaded', populateTable);
