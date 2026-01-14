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
  toggleMapOptions();
  fetchMapProviders();
  fetchStations();
};
