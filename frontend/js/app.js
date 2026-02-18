let currentIndex = -1;

function toggleMapOptions() {
  const mapOptions = document.getElementById("map-options");
  const enabled = document.getElementById("generate-map").checked;
  fetchMapProviders();

  mapOptions.style.display = enabled ? "block" : "none";
  mapOptions.querySelectorAll("input, select, textarea, button").forEach((el) => (el.disabled = !enabled));
}

function isContrasting(color) {
  const r = parseInt(color.slice(1, 3), 16);
  const g = parseInt(color.slice(3, 5), 16);
  const b = parseInt(color.slice(5, 7), 16);

  const brightness = (r * 299 + g * 587 + b * 114) / 1000;
  return brightness < 157;
}

function isVibrant(color) {
  const r = parseInt(color.slice(1, 3), 16);
  const g = parseInt(color.slice(3, 5), 16);
  const b = parseInt(color.slice(5, 7), 16);

  const maxChannel = Math.max(r, g, b);
  const minChannel = Math.min(r, g, b);
  const saturation = maxChannel ? (maxChannel - minChannel) / maxChannel : 0;

  return saturation > 0.2 && saturation < 0.9;
}

function generateContrastingVibrantColor(mode) {
  const chars = "0123456789abcdef";
  for (;;) {
    let color = "#";
    for (let i = 0; i < 6; i++) {
      color += chars[Math.floor(Math.random() * chars.length)];
    }
    if (mode == "dark") {
      if (!isContrasting(color) && isVibrant(color)) return color;
    } else {
      if (isContrasting(color) && isVibrant(color)) return color;
    }
  }
}

function changeColor() {
  const color = document.getElementById("color").value;
  document.documentElement.style.setProperty("--acc-color", color);
}

function updateHighlight(listEl) {
  const items = listEl.querySelectorAll("li");
  items.forEach((el, i) => {
    const active = i === currentIndex;
    el.classList.toggle("active", active);
    el.setAttribute("aria-selected", active ? "true" : "false");
  });
  if (currentIndex >= 0 && items[currentIndex]) {
    items[currentIndex].scrollIntoView({ block: "nearest" });
  }
}

function clearSuggestions() {
  const dataList = document.getElementById("station_datalist");
  dataList.innerHTML = "";
  dataList.style.display = "none";
  currentIndex = -1;
}

function fillSuggestion(event) {
  const stationInput = document.getElementById("station_name");

  const text = event.currentTarget.textContent;
  stationInput.value = text;

  clearSuggestions();
  stationInput.focus();
}

async function fetchStations() {
  const response = await fetch("/api/stations");
  const stations = await response.json();
  window.stations = stations.stations;
}

async function fetchSuggestions() {
  const inputEl = document.getElementById("station_name");
  const dataList = document.getElementById("station_datalist");
  const q = inputEl.value;

  if (q.length < 3) {
    clearSuggestions();
    return;
  }

  const response = await fetch(`/api/stations?query=${encodeURIComponent(q)}`);
  const stations = await response.json();

  dataList.innerHTML = "";
  dataList.setAttribute("role", "listbox");

  if (stations.total === 0 || (stations.total === 1 && stations.stations[0] === q)) {
    dataList.style.display = "none";
    currentIndex = -1;
    return;
  }

  stations.stations.forEach((station, i) => {
    const li = document.createElement("li");
    li.textContent = station;
    li.setAttribute("role", "option");
    li.setAttribute("aria-selected", "false");
    li.addEventListener("mousedown", fillSuggestion);

    dataList.appendChild(li);
  });

  dataList.style.display = "block";
  currentIndex = -1;
  updateHighlight(dataList);
}

function select(event) {
  const dataList = document.getElementById("station_datalist");
  const items = dataList.querySelectorAll("li");

  if (!["ArrowDown", "ArrowUp", "Enter", "Escape"].includes(event.key)) return;

  const listVisible = dataList.style.display !== "none" && items.length > 0;

  if (!listVisible) {
    if (event.key === "ArrowDown") {
      if (document.getElementById("station_name").value.length >= 3) {
        fetchSuggestions();
      }
    }
    return;
  }

  event.preventDefault();
  event.stopPropagation();

  if (event.key === "ArrowDown") {
    currentIndex = Math.min(currentIndex + 1, items.length - 1);
    updateHighlight(dataList);
    return;
  }

  if (event.key === "ArrowUp") {
    currentIndex = Math.max(currentIndex - 1, 0);
    updateHighlight(dataList);
    return;
  }

  if (event.key === "Enter") {
    if (currentIndex >= 0) {
      items[currentIndex].dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
    }
    return;
  }

  if (event.key === "Escape") {
    clearSuggestions();
    return;
  }
}

function removeOptions(selectElement) {
  var i,
    L = selectElement.options.length - 1;
  for (i = L; i >= 0; i--) {
    selectElement.remove(i);
  }
}

async function fetchMapProviders() {
  const response = await fetch("/api/map-providers");
  const providers = await response.json();
  const providerSelect = document.getElementById("map_provider");
  removeOptions(providerSelect);
  providers.map_providers.forEach((provider) => {
    const option = document.createElement("option");
    option.value = provider;
    option.textContent = provider;
    providerSelect.add(option);
  });
}

async function pollForDownload(dl) {
  const pollInterval = 3000;

  while (true) {
    const res = await fetch(`/api/status?dl=${dl}`);

    if (res.status === 202) {
      await new Promise((r) => setTimeout(r, pollInterval));
      continue;
    }

    if (res.status === 200) {
      const data = await res.json();
      window.location.href = `/api/download?dl=${data.download}`;
      break;
    }

    const err = await res.json();
    alert(err.detail || "Download failed");
    break;
  }
}

async function handleFormSubmit(event) {
  event.preventDefault();
  if (!validateForm()) return;

  const formData = new FormData(document.getElementById("schedule-form"));
  // disable form after submit to prevent api spam
  const submitButton = document.getElementById("submit");
  const loader = document.getElementById("loader");
  submitButton.disabled = true;
  loader.style.display = "flex";

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      body: formData,
    });

    if (response.ok) {
      const data = await response.json();
      await pollForDownload(data.download);
    } else {
      const error = await response.json();
      alert("Error generating timetable: " + error.detail);
    }
  } finally {
    submitButton.disabled = false;
    loader.style.display = "none";
  }
}

function validateForm() {
  const stationInput = document.getElementById("station_name");
  const stationName = stationInput.value;

  if (!Array.isArray(window.stations)) {
    console.warn("Stations not loaded yet; skipping strict validation.");
    return true;
  }

  if (!window.stations.includes(stationName)) {
    alert("Please enter a valid station.");
    return false;
  }
  return true;
}

function darkMode() {
  const colorPicker = document.getElementById("color");
  const color = generateContrastingVibrantColor("dark");
  document.documentElement.style.setProperty("--acc-color", color);
  document.documentElement.style.setProperty("--text-color", "#bfbfbf");
  document.documentElement.style.setProperty("--alt-color", "black");
  document.documentElement.style.setProperty("--bg-color", "#0f0f0f");
  document.documentElement.style.setProperty("--overlay-color", "#0f0f0fcc");
  colorPicker.value = color;
}
function lightMode() {
  const colorPicker = document.getElementById("color");
  const color = generateContrastingVibrantColor();
  document.documentElement.style.setProperty("--acc-color", color);
  document.documentElement.style.setProperty("--text-color", "black");
  document.documentElement.style.setProperty("--alt-color", "white");
  document.documentElement.style.setProperty("--bg-color", "#fafafa");
  document.documentElement.style.setProperty("--overlay-color", "#fafafacc");
  colorPicker.value = color;
}

function darkModeToggle(mode) {
  const switchState = document.getElementById("dark-mode-switch-check");
  if (mode == "dark") {
    darkMode();
    switchState.checked = true;
  } else if (mode == "light") {
    lightMode();
    switchState.checked = false;
  } else {
    if (switchState.checked) {
      darkMode();
    } else {
      lightMode();
    }
  }
}

function detectDarkMode() {
  if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
    darkModeToggle("dark");
  } else {
    darkModeToggle("light");
  }
}

document.getElementById("dark-mode-switch-check").addEventListener("change", darkModeToggle);
document.getElementById("schedule-form").addEventListener("submit", handleFormSubmit);
document.getElementById("generate-map").addEventListener("change", toggleMapOptions);
document.getElementById("station_name").addEventListener("input", fetchSuggestions);
document.getElementById("station_name").addEventListener("keydown", select);
document.getElementById("color").addEventListener("change", changeColor);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (event) => {
  const newColorScheme = event.matches ? "dark" : "light";
  darkModeToggle(newColorScheme);
});

detectDarkMode();

window.onload = function () {
  toggleMapOptions();
  fetchMapProviders();
  fetchStations();
};
