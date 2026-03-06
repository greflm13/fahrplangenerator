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

function clearSuggestions(type) {
  const dataList = document.getElementById(type + "_datalist");
  dataList.innerHTML = "";
  dataList.style.display = "none";
  currentIndex = -1;
}

function fillSuggestion(event) {
  type = event.target.parentElement.id.split("_")[0];
  const input = document.getElementById(type + "_name");

  const text = event.currentTarget.textContent;
  input.value = text;

  clearSuggestions(type);
  input.focus();
}

async function fetchStations() {
  const response = await fetch("/api/stations");
  const stations = await response.json();
  window.stations = stations.stations;
}

async function fetchStationSuggestions() {
  await fetchSuggestions("stations");
}

async function fetchAgenciesSuggestions() {
  await fetchSuggestions("agencies");
}

async function fetchRoutesSuggestions() {
  await fetchSuggestions("routes");
}

async function fetchSuggestions(type) {
  const inputEl = document.getElementById(type + "_name");
  const dataList = document.getElementById(type + "_datalist");
  const q = inputEl.value;

  if (type != "agencies" && q.length < 3) {
    clearSuggestions(type);
    return;
  } else {
    const agencyId = document.getElementById("agencies_name").text;
  }

  if (type != "agencies") {
    const response = await fetch(`/api/` + type + `?query=${encodeURIComponent(q)}`);
  } else {
    const response = await fetch(`/api/` + type + `?query=${encodeURIComponent(q)}` + `&agency=${encodeURIComponent(agencyId)}`);
  }
  const res = await response.json();

  dataList.innerHTML = "";
  dataList.setAttribute("role", "listbox");

  if (res.total === 0 || (res.total === 1 && res.data[0] === q)) {
    dataList.style.display = "none";
    currentIndex = -1;
    return;
  }

  res.data.forEach((el, i) => {
    const li = document.createElement("li");
    li.textContent = el;
    li.setAttribute("role", "option");
    li.setAttribute("aria-selected", "false");
    dataList.appendChild(li);
    li.addEventListener("mousedown", fillSuggestion);
  });

  dataList.style.display = "block";
  currentIndex = -1;
  updateHighlight(dataList);
}

function select(event) {
  type = event.target.id.split("_")[0];
  const dataList = document.getElementById(type + "_datalist");
  const items = dataList.querySelectorAll("li");

  if (!["ArrowDown", "ArrowUp", "Enter", "Escape"].includes(event.key)) return;

  const listVisible = dataList.style.display !== "none" && items.length > 0;

  if (!listVisible) {
    if (event.key === "ArrowDown") {
      if (document.getElementById(type + "_name").value.length >= 3) {
        fetchSuggestions(type);
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
    clearSuggestions(type);
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

stationsEl = document.getElementById("stations_name");
agenciesEl = document.getElementById("agencies_name");
routesEl = document.getElementById("routes_name");
mapEl = document.getElementById("generate-map");

document.getElementById("dark-mode-switch-check").addEventListener("change", darkModeToggle);
document.getElementById("schedule-form").addEventListener("submit", handleFormSubmit);
if (mapEl != null) {
  mapEl.addEventListener("change", toggleMapOptions);
}
if (stationsEl != null) {
  stationsEl.addEventListener("input", fetchStationSuggestions);
  stationsEl.addEventListener("keydown", select);
}
if (agenciesEl != null) {
  agenciesEl.addEventListener("input", fetchAgenciesSuggestions);
  agenciesEl.addEventListener("keydown", select);
}
if (routesEl != null) {
  routesEl.addEventListener("input", fetchRoutesSuggestions);
  routesEl.addEventListener("keydown", select);
}

document.getElementById("color").addEventListener("change", changeColor);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (event) => {
  const newColorScheme = event.matches ? "dark" : "light";
  darkModeToggle(newColorScheme);
});

detectDarkMode();

window.onload = function () {
  if (mapEl != null) {
    toggleMapOptions();
  }
  fetchMapProviders();
  if (stationsEl != null) {
    fetchStations();
  }
};
