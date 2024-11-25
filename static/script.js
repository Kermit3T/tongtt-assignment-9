document.getElementById("experiment-form").addEventListener("submit", async function(event) {
    event.preventDefault();

    // Get form elements
    const form = event.target;
    const submitButton = form.querySelector('.submit-button');
    const loadingDiv = document.querySelector('.loading');
    const formElements = form.elements;
    
    // Validation
    const activation = document.getElementById("activation").value;
    const lrInput = document.getElementById("lr");
    const lr = parseFloat(lrInput.value);
    const stepNum = parseInt(document.getElementById("step_num").value);

    // Validate learning rate
    if (lr < 0.001 || lr > 1 || isNaN(lr)) {
        lrInput.setCustomValidity("Please enter a value between 0.001 and 1");
        lrInput.reportValidity();
        return;
    } else {
        lrInput.setCustomValidity("");
    }

    // Show loading state
    submitButton.disabled = true;
    loadingDiv.style.display = 'block';
    
    // Disable all form inputs during processing
    for (let element of formElements) {
        element.disabled = true;
    }

    try {
        const response = await fetch("/run_experiment", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ 
                activation: activation, 
                lr: lr, 
                step_num: stepNum 
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'An error occurred');
        }

        const data = await response.json();
        
        // Show results
        const resultsDiv = document.getElementById("results");
        const resultImg = document.getElementById("result_gif");
        
        if (data.result_gif) {
            // Add timestamp to prevent caching
            resultImg.src = `/${data.result_gif}?t=${new Date().getTime()}`;
            resultsDiv.style.display = "block";
            // Scroll to results
            resultsDiv.scrollIntoView({ behavior: 'smooth' });
        } else {
            throw new Error("No visualization generated");
        }
    } catch (error) {
        console.error("Error:", error);
        alert(error.message || "An error occurred while running the experiment. Please try again.");
    } finally {
        // Hide loading state and re-enable form
        loadingDiv.style.display = 'none';
        submitButton.disabled = false;
        
        // Re-enable all form inputs
        for (let element of formElements) {
            element.disabled = false;
        }
    }
});

// Add input validation for learning rate
document.getElementById("lr").addEventListener("input", function(event) {
    const value = parseFloat(event.target.value);
    if (value < 0.001 || value > 1 || isNaN(value)) {
        event.target.setCustomValidity("Please enter a value between 0.001 and 1");
    } else {
        event.target.setCustomValidity("");
    }
});

// Add clear results functionality
document.querySelector('.clear-button').addEventListener('click', function() {
    const resultsDiv = document.getElementById("results");
    if (resultsDiv) {
        resultsDiv.style.display = "none";
        const resultImg = document.getElementById("result_gif");
        if (resultImg) {
            resultImg.src = "";
            // Force browser to clear the image from memory
            resultImg.removeAttribute('src');
        }
        // Scroll back to top
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    }
});