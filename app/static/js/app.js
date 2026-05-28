(function () {
    function init() {
        const tabFile = document.getElementById("tab-file");
        const tabUrl = document.getElementById("tab-url");
        const contentFile = document.getElementById("content-file");
        const contentUrl = document.getElementById("content-url");
        const modelSelect = document.getElementById("model-select");
        const fileInput = document.getElementById("file-input");
        const urlInput = document.getElementById("url-input");
        const classifyBtn = document.getElementById("btn-classify");
        const errorMessage = document.getElementById("error-message");

        const previewCard = document.getElementById("preview-card");
        const previewImage = document.getElementById("preview-image");
        const clearPreview = document.getElementById("clear-preview");

        const resultCard = document.getElementById("result-card");
        const outBreed = document.getElementById("out-breed");
        const outProb = document.getElementById("out-prob");
        const outTime = document.getElementById("out-time");
        const outModel = document.getElementById("out-model");
        const outTop5 = document.getElementById("out-top5");
        const mainFill = document.getElementById("main-fill");

        if ([tabFile, tabUrl, contentFile, contentUrl, modelSelect, fileInput, urlInput, classifyBtn, errorMessage, previewCard, previewImage, clearPreview, resultCard, outBreed, outProb, outTime, outModel, outTop5, mainFill].some((n) => !n)) {
            return;
        }

        let mode = "file";
        let objectUrl = "";

        function showError(msg) { errorMessage.textContent = msg || ""; }

        function clearPreviewUI() {
            if (objectUrl) {
                URL.revokeObjectURL(objectUrl);
                objectUrl = "";
            }
            previewImage.src = "";
            previewCard.classList.add("hidden");
        }

        function setMode(next) {
            mode = next;
            tabFile.classList.toggle("active", next === "file");
            tabUrl.classList.toggle("active", next === "url");
            contentFile.classList.toggle("active", next === "file");
            contentUrl.classList.toggle("active", next === "url");
            showError("");
        }

        function setPreviewFromFile(file) {
            if (!file) return;
            clearPreviewUI();
            objectUrl = URL.createObjectURL(file);
            previewImage.src = objectUrl;
            previewCard.classList.remove("hidden");
        }

        async function setPreviewFromUrl(url) {
            if (!url) return;
            previewImage.src = url;
            previewCard.classList.remove("hidden");
        }

        function renderResult(data) {
            outBreed.textContent = data.breed;
            const pct = (data.probability * 100).toFixed(1);
            outProb.textContent = `${pct}%`;
            outTime.textContent = data.inference_time_ms;
            outModel.textContent = data.model_used;
            mainFill.style.width = `${pct}%`;

            outTop5.innerHTML = "";
            data.top5.forEach((p, i) => {
                const rowPct = (p.confidence * 100).toFixed(1);
                const li = document.createElement("li");
                li.innerHTML = `
                    <span>${i + 1}. ${p.label}</span>
                    <div class="row-meter"><div class="row-fill" style="width:${rowPct}%"></div></div>
                    <strong>${rowPct}%</strong>
                `;
                outTop5.appendChild(li);
            });

            resultCard.classList.remove("hidden");
        }

        tabFile.addEventListener("click", () => setMode("file"));
        tabUrl.addEventListener("click", () => setMode("url"));

        fileInput.addEventListener("change", () => {
            const file = fileInput.files && fileInput.files[0];
            if (file) setPreviewFromFile(file);
        });

        urlInput.addEventListener("change", () => {
            const url = urlInput.value.trim();
            if (url) setPreviewFromUrl(url);
        });

        clearPreview.addEventListener("click", clearPreviewUI);

        classifyBtn.addEventListener("click", async () => {
            showError("");
            resultCard.classList.add("hidden");

            const model = modelSelect.value;
            try {
                let response;
                if (mode === "file") {
                    const file = fileInput.files && fileInput.files[0];
                    if (!file) {
                        showError("Selecciona una imagen.");
                        return;
                    }
                    const fd = new FormData();
                    fd.append("file", file);
                    response = await fetch(`/predict/file?model=${encodeURIComponent(model)}`, {
                        method: "POST",
                        body: fd,
                    });
                } else {
                    const url = urlInput.value.trim();
                    if (!url) {
                        showError("Ingresa una URL.");
                        return;
                    }
                    await setPreviewFromUrl(url);
                    response = await fetch(`/predict/url?model=${encodeURIComponent(model)}`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ url }),
                    });
                }

                const data = await response.json();
                if (!response.ok) {
                    showError(data.detail || "Error de prediccion");
                    return;
                }
                renderResult(data);
            } catch {
                showError("Error de conexion con la API");
            }
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();

