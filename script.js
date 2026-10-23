function formValues(form) {
  const values = {};
  new FormData(form).forEach((value, key) => values[key] = Number(value));
  return values;
}

async function sendPrediction(formId, endpoint, resultId, mode) {
  const form = document.getElementById(formId);
  const result = document.getElementById(resultId);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    result.textContent = "Making prediction...";
    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formValues(form))
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "The prediction failed.");
      if (mode === "regression") {
        result.textContent = `Predicted average rating: ${data.predicted_average_rating} / 5`;
      } else {
        result.textContent = `Predicted category: ${data.category}`;
      }
    } catch (error) {
      result.textContent = `Error: ${error.message}. Make sure the models are trained first.`;
    }
  });
}

sendPrediction("regression-form", "/predict/regression", "regression-result", "regression");
sendPrediction("classification-form", "/predict/classification", "classification-result", "classification");
