// JavaScript logic for map.html

const map = L.map("map", {
    maxZoom: 19.9  // This is the max possible zoom level supported by ArcGIS for this location
});

let hoverMarker = null;
let uploadedTrajectoryLayer = null;
let lapTrajectoryLayer = null;

L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    {
        maxZoom: 19.9,
    }
).addTo(map);

// Start Position Marker (red square with black border)
const flagIcon = L.divIcon({
  className: 'custom-square',
  iconSize: [10, 10],
  iconAnchor: [5, 5],
  html: `
    <div style="
      width: 10px;
      height: 10px;
      background-color: #ff0000;
      border: 1px solid black;
    "></div>`
});

// Segment Points Marker (blue square with black border)
const segmentIcon = L.divIcon({
  className: 'custom-square',
  iconSize: [10, 10],
  iconAnchor: [5, 5],
  html: `
    <div style="
      width: 10px;
      height: 10px;
      background-color: #007bff;
      border: 1px solid black;
    "></div>`
});

// Load track data
fetch("/get_track_data")
    .then((response) => response.json())
    .then((data) => {
        // Draw segments
        data.segments.forEach((segment) => {
        const line = L.polyline(segment.points, {
            color: segment.color,
            weight: 4,
            opacity: 0.8,
        }).addTo(map);

        // Select zones
        line.on("click", () => {
            window.parent.postMessage(
            {
                type: "segment_click",
                zone: segment.zone,
            },
            "*"
            );
        });

        // Data displayed on hover
        line.bindTooltip(`${segment.zone}`);
    });

    // Add markers
    L.marker(data.start_finish, { icon: flagIcon })
        .addTo(map)
        .bindPopup("Start/Finish");

        data.segment_points.forEach((point, i) => {
        L.marker(point, { icon: segmentIcon })
            .addTo(map)
            .bindPopup(`Segment Point ${i + 1}`);
    });

    // Draw bridge lines
    if (data.bridges) {
        data.bridges.forEach((bridge, idx) => {
            const bridgeLine = L.polyline(bridge.points, {
                color: 'black',
                weight: 3,
                opacity: 1.0
            }).addTo(map);

            // Bind tooltip (hover text)
            const label = bridge.label || `Bridge ${idx + 1}`;
            bridgeLine.bindTooltip(label, { permanent: false, direction: "top" });
        });
    }

    // Set default view on map
    const bounds = L.latLngBounds(data.segments.flatMap((s) => s.points));
    map.fitBounds(bounds);  // Auto-zoom to fit full track

    // Get selected segments information to be highlighted
    window.parent.trackMap = map;
    window.addEventListener("message", function (event) {
        // Highlight selected segment
        if (event.data.type === "highlight_segment") {
            if (window.highlightLayer) {
            map.removeLayer(window.highlightLayer);
            window.highlightLayer = null;
            }

            if (event.data.points) {
            window.highlightLayer = L.polyline(event.data.points, {
                color: "yellow",
                weight: 5,
                opacity: 0.9,
            }).addTo(map);

            // Tooltip for combined zone
            const label = event.data.label || "Selected Segments";
            window.highlightLayer.bindTooltip(label, { sticky: true });
            }
        }

        // Highlight the point from hover on plot
        if (event.data.type === "highlight_point") {
            const { lat, lon } = event.data;

            if (!hoverMarker) {
                const divIcon = L.divIcon({
                    className: "",
                    html:
                      '<div style="width:16px;height:16px;background:rgba(255,0,0,0.9);border-radius:50%;position:relative;">' +
                      '<div style="position:absolute;width:100%;height:100%;border:2px solid rgba(255,0,0,0.5);border-radius:50%;' +
                      'animation:pulse 1s infinite ease-in-out;top:-4px;left:-4px;"></div></div>',
                    iconSize: [16, 16],
                    iconAnchor: [8, 8],
                });
                hoverMarker = L.marker([lat, lon], { icon: divIcon }).addTo(map);
            } else {
                hoverMarker.setLatLng([lat, lon]);
            }
        }

        // Mark trajectory on map
        if (event.data.type === "draw_uploaded_trajectory") {
        if (uploadedTrajectoryLayer) {
            map.removeLayer(uploadedTrajectoryLayer);
        }

        uploadedTrajectoryLayer = L.polyline(event.data.points, {
            color: "blue",
            weight: 1,
            opacity: 0.8
            // dashArray: "4, 6"
        }).addTo(map);
        }  
    });
});