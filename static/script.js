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

        // Append cells to row
        row.appendChild(modelCell);
        row.appendChild(paramCell);
        row.appendChild(map50_95Cell);
        row.appendChild(map50Cell);
        row.appendChild(map75Cell);
        row.appendChild(smallMap50_95Cell);
        row.appendChild(mediumMap50_95Cell);
        row.appendChild(largeMap50_95Cell);

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
