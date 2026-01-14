function toggleMapOptions() {
  const mapOptions = document.getElementById("map-options");
  if (document.getElementById("generate-map").checked) {
    mapOptions.style.display = "block";
  } else {
    mapOptions.style.display = "none";
  }
}

async function fetchStations(query = "") {
  if (query.length < 3) {
    return;
  }
  const response = await fetch(`/api/stations?query=${encodeURIComponent(query)}`);
  const stations = await response.json();
  const dataList = document.getElementById("station_datalist");
  dataList.innerHTML = ""; // Clear existing options
  stations.stations.forEach((station) => {
    const option = document.createElement("option");
    option.value = station;
    dataList.appendChild(option);
  });
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

document.getElementById("generate-map").addEventListener("change", toggleMapOptions);

window.onload = function () {
  toggleMapOptions();
  fetchMapProviders();
  fetchStations();
};
