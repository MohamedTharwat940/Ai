document.getElementById("predictionForm").addEventListener("submit", async (event) => {
    event.preventDefault(); // Prevent form submission

    // Collect input values
    const avg = parseFloat(document.getElementById("Avg").value);
    const toapp = parseFloat(document.getElementById("Toapp").value);
    const toweb = parseFloat(document.getElementById("Toweb").value);
    const lomember = parseFloat(document.getElementById("Lomember").value);

    try {
        // Send data to Flask backend
        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                Avg: avg,
                Toapp: toapp,
                Toweb: toweb,
                Lomember: lomember,
            }),
        });

        const result = await response.json();

        if (result.error) {
            document.getElementById("result").innerHTML = `<p>Error: ${result.error}</p>`;
        } else {
            // Display the predictions
            document.getElementById("result").innerHTML = `
                <p><strong>Linear Regression:</strong> $${result['Linear Regression Prediction']}</p>
                <p><strong>KNN:</strong> $${result['KNN Prediction']}</p>
            `;
        }
    } catch (error) {
        document.getElementById("result").innerHTML = `<p>Error: ${error.message}</p>`;
    }
});
