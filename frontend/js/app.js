function toggleMapOptions() {
  const mapOptions = document.getElementById("map-options");
  if (document.getElementById("generate-map").checked) {
    mapOptions.style.display = "block";
  } else {
    mapOptions.style.display = "none";
  }
}

async function fetchStations() {
  const response = await fetch("/stations");
  const stations = await response.json();
  const stationSelect = document.getElementById("station-select");
  stations.stations.forEach((station) => {
    const option = document.createElement("option");
    option.value = station;
    option.text = station;
    stationSelect.add(option);
  });
}

async function fetchMapProviders() {
  const response = await fetch("/map-providers");
  const providers = await response.json();
  const providerSelect = document.getElementById("map-provider-select");
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
