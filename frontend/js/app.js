function toggleMapOptions() {
  const mapOptions = document.getElementById("map-options");
  if (document.getElementById("generate-map").checked) {
    mapOptions.style.display = "block";
    mapOptions.disabled = false;
  } else {
    mapOptions.style.display = "none";
    mapOptions.disabled = true;
  }
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

function generateContrastingVibrantColor() {
  const chars = "0123456789abcdef";

  while (true) {
    let color = "#";
    for (let i = 0; i < 6; i++) {
      color += chars[Math.floor(Math.random() * chars.length)];
    }

    if (isContrasting(color) && isVibrant(color)) {
      return color;
    }
  }
}

function generateColor() {
  const color = generateContrastingVibrantColor();
  const colorPicker = document.getElementById("color");
  document.documentElement.style.setProperty("--acc-color", color);
  colorPicker.value = color;
}

function changeColor() {
  const color = document.getElementById("color").value;
  document.documentElement.style.setProperty("--acc-color", color);
}

async function fetchStations(query = "") {
  if (query === "") {
    const response = await fetch("/api/stations");
    const stations = await response.json();
    window.stations = stations.stations;
  } else {
    const dataList = document.getElementById("station_datalist");
    dataList.innerHTML = ""; // Clear existing options
    if (query.length < 3) {
      return;
    }
    const response = await fetch(
      `/api/stations?query=${encodeURIComponent(query)}`
    );
    const stations = await response.json();
    stations.stations.forEach((station) => {
      const option = document.createElement("option");
      option.value = station;
      dataList.appendChild(option);
    });
  }
}

async function fetchMapProviders() {
  const response = await fetch("/api/map-providers");
  const providers = await response.json();
  const providerSelect = document.getElementById("map_provider");
  providers.map_providers.forEach((provider) => {
    const option = document.createElement("option");
    option.value = provider;
    option.text = provider;
    providerSelect.add(option);
  });
}

async function handleFormSubmit(event) {
  event.preventDefault();
  if (!validateForm()) {
    return;
  }

  const formData = new FormData(document.getElementById("schedule-form"));
  // disable form after submit to prevent api spam
  const submitButton = document.getElementById("submit");
  const loader = document.getElementById("loader");
  submitButton.disabled = true;
  loader.style.display = "flex";

  const response = await fetch("/api/generate", {
    method: "POST",
    body: formData,
  });

  submitButton.disabled = false;
  loader.style.display = "none";

  if (response.ok) {
    const data = await response.json();
    window.location.href = `/api/download?dl=${data.download}`;
  } else {
    const error = await response.json();
    alert("Error generating timetable: " + error.detail);
  }
}

function validateForm() {
  const stationInput = document.getElementById("station_name");
  const stationName = stationInput.value;

  if (!window.stations.includes(stationName)) {
    alert("Please enter a valid station.");
    return false;
  } else {
    return true;
  }
}

document
  .getElementById("generate-map")
  .addEventListener("change", toggleMapOptions);

window.onload = function () {
  generateColor();
  toggleMapOptions();
  fetchMapProviders();
  fetchStations();
};
