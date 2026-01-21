  // JavaScript logic for index.html

  let currentZone = null;
  let currentStartZone = null;
  let currentEndZone = null;
  let allZones = [];
  let selectedColumnKeywords = [];
  let availablePlotColumns = {}; // keyword -> column list
  let userSelectedPlots = new Set();
  let showAllByDefault = true;

  const startSelect = document.getElementById("start-segment-select");
  const endSelect = document.getElementById("end-segment-select");
  const applyRangeButton = document.getElementById("apply-range");

  // Enable or Disable Apply button based on dropdown values
  function updateApplyButtonState() {
    applyRangeButton.disabled = !(startSelect.value && endSelect.value);
  }

  loadZoneDropdowns();
  updateUploadedFilesList();
  initializeFromExistingData();


  // 0. Initializer Function
  function initializeFromExistingData() {
    // 1. Restore lap selection
    const savedLap = localStorage.getItem("selectedLap");
    fetch("/get_available_laps")
      .then(res => res.json())
      .then(laps => {
        const select = document.getElementById("lap-selector");
        select.innerHTML = "";

        const defaultOpt = document.createElement("option");
        defaultOpt.value = "";
        defaultOpt.textContent = "All Laps (full data)";
        select.appendChild(defaultOpt);

        if (!laps.length) {
          document.getElementById("lap-selector-wrapper").style.display = "none";
          return;
        }

        laps.filter(lap => lap !== -1 && lap !== "-1").forEach(lap => {
          const opt = document.createElement("option");
          opt.value = lap;
          opt.textContent = "Lap " + lap;
          select.appendChild(opt);
        });

        document.getElementById("lap-selector-wrapper").style.display = "block";

        if (savedLap) {
          select.value = savedLap;
        }
      });

    // 2. Restore column dropdown & selection
    fetch("/get_combined_columns")
      .then(res => res.json())
      .then(columns => {
        const dropdown = document.getElementById("column-dropdown");
        dropdown.innerHTML = '<option disabled selected>Choose column...</option>';
        columns.forEach(col => {
          const option = document.createElement("option");
          option.value = col.value;
          option.textContent = col.label;
          dropdown.appendChild(option);
        });

        const savedColumns = JSON.parse(localStorage.getItem("selectedColumns") || "[]");
        savedColumns.forEach(val => {
          const opt = Array.from(dropdown.options).find(o => o.value === val);
          if (opt) opt.selected = true;
        });

        selectedColumnKeywords = savedColumns;
      });

    // 3. Restore showAllByDefault and userSelectedPlots
    showAllByDefault = JSON.parse(localStorage.getItem("showAllByDefault") || "true");
    const savedUserPlots = JSON.parse(localStorage.getItem("userSelectedPlots") || "[]");
    userSelectedPlots = new Set(savedUserPlots);

    // 4. Restore trajectory
    fetch("/get_uploaded_trajectory")
      .then(res => res.json())
      .then(data => {
        if (data.lat && data.lon && data.lat.length > 0) {
          const points = data.lat.map((lat, i) => [lat, data.lon[i]]);
          document.getElementById("track-map").contentWindow.postMessage({
            type: "draw_uploaded_trajectory",
            points
          }, "*");
        }
      });

    // 5. Restore zone/segment selection
    const segmentStart = localStorage.getItem("segmentStart");
    const segmentEnd = localStorage.getItem("segmentEnd");
    const clickedZone = localStorage.getItem("clickedZone");

    const payload = {};
    if (clickedZone) {
      payload.zone = clickedZone;
      currentZone = clickedZone;
      currentStartZone = null;
      currentEndZone = null;
    } else if (segmentStart && segmentEnd) {
      payload.start_zone = segmentStart;
      payload.end_zone = segmentEnd;

      currentZone = null;
      currentStartZone = segmentStart;
      currentEndZone = segmentEnd;

      startSelect.value = segmentStart;
      endSelect.value = segmentEnd;
      updateApplyButtonState();
    }

    if (savedLap) {
      payload.lap = savedLap;
    }

    // 6. Fetch & render initial plot + map
    fetch("/segment_data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(res => res.json())
      .then(data => {
        updatePlotsAndMap(data);
        if (selectedColumnKeywords.length > 0) {
          const lap = localStorage.getItem("selectedLap");
          const preloadPromises = selectedColumnKeywords.map(kw => {
            const queryParams = new URLSearchParams({ keyword: kw });
            if (lap) queryParams.set("lap", lap);
            if (currentZone) {
              queryParams.set("zone", currentZone);
            } else if (currentStartZone && currentEndZone) {
              queryParams.set("start_zone", currentStartZone);
              queryParams.set("end_zone", currentEndZone);
            }

            return fetch(`/get_column_data?${queryParams.toString()}`)
              .then(res => res.json())
              .then(data => {
                availablePlotColumns[kw] = Object.keys(data.cols || {});
              });
          });

          Promise.all(preloadPromises).then(() => {
            plotCurrentColumn();
          });
        }
      });
  }


  // 1. Handle segment click from map iframe
  window.addEventListener("message", function (event) {
  if (event.data.type === "segment_click") {
    currentZone = event.data.zone;
    currentStartZone = null;
    currentEndZone = null;

    // Store clicked zone in localStorage
    localStorage.setItem("clickedZone", currentZone);
    localStorage.removeItem("segmentStart");
    localStorage.removeItem("segmentEnd");

    startSelect.selectedIndex = 0;
    endSelect.selectedIndex = 0;
    updateApplyButtonState();

    const lap = document.getElementById("lap-selector")?.value || null;

    fetch("/segment_data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ zone: currentZone, lap })
    })
      .then(res => res.json())
      .then(data => {
        updatePlotsAndMap(data);

        // Preload column data with correct lap and zone
        const dropdown = document.getElementById("column-dropdown");
        const selectedKeywords = Array.from(dropdown.selectedOptions)
          .map(opt => opt.value)
          .filter(v => v && v !== "Choose column...");

        if (selectedKeywords.length === 0) return;

        const preloadPromises = selectedKeywords.map(kw => {
          const queryParams = new URLSearchParams({ keyword: kw });
          if (lap) queryParams.set("lap", lap);
          if (currentZone) queryParams.set("zone", currentZone);

          return fetch(`/get_column_data?${queryParams.toString()}`)
            .then(res => res.json())
            .then(data => {
              availablePlotColumns[kw] = Object.keys(data.cols || {});
            });
        });

        Promise.all(preloadPromises).then(() => {
          plotCurrentColumn();
        });
      });
  }
});


  // 2. Update plots and map display
  function updatePlotsAndMap(data) {
    document.getElementById("chart-title").innerText = "Plot";
    document.getElementById("segment-title").innerText = `Zone: ${data.zone}`;

    const layout = {
      margin: { t: 20 },
      xaxis: { title: "Distance (m)" },
      yaxis: { title: "Velocity (m/s)", side: "left" },
      yaxis2: {
        title: "Acceleration (m/s²)",
        side: "right",
        overlaying: "y",
        showgrid: false
      },
      legend: { orientation: "h", x: 0, y: 1.15 },
      hovermode: "x unified",
      hoverlabel: { bgcolor: "#fff", font: { color: "#000" } }
    };

    const chartContainer = document.getElementById("chart-container");
    chartContainer.innerHTML = ""; // Clear previous plot

    if (data.x && data.cols) {
      const traces = Object.entries(data.cols).map(([colName, y], i) => ({
        x: data.x,
        y,
        name: colName,
        yaxis: i % 2 === 0 ? "y" : "y2",
        type: "scatter",
        mode: "lines"
      }));

      const chartDiv = document.createElement("div");
      chartDiv.style.height = "360px";
      chartContainer.appendChild(chartDiv);

      Plotly.newPlot(chartDiv, traces, layout);

      chartDiv.on("plotly_hover", ev => {
        const idx = ev.points[0].pointIndex;
        const lat = data.lat[idx];
        const lon = data.lon[idx];

        document.getElementById("track-map").contentWindow.postMessage(
          { type: "highlight_point", lat, lon },
          "*"
        );
      });
    }

    if (data.highlight_points && data.highlight_points.length > 0) {
      document.getElementById("track-map").contentWindow.postMessage({
        type: "highlight_segment",
        points: data.highlight_points,
        label: data.highlight_label || `Zone: ${data.zone}`
      }, "*");
    }
  }


  // 3. Build Start and End dropdowns
  function rebuildZoneDropdowns(zones) {
    zones.sort((a, b) => Number(a.match(/\d+/)) - Number(b.match(/\d+/)));

    startSelect.innerHTML =
      '<option value="" disabled selected>Select Start Segment</option>';
    endSelect.innerHTML =
      '<option value="" disabled selected>Select End Segment</option>';

    zones.forEach(z => {
      startSelect.appendChild(new Option(z, z));
      endSelect.appendChild(new Option(z, z));
    });

    startSelect.addEventListener("change", updateApplyButtonState);
    endSelect.addEventListener("change", updateApplyButtonState);
  }


  // 4. Load zone labels from server
  function loadZoneDropdowns() {
    fetch("/get_zone_labels")
      .then(res => res.json())
      .then(zones => {
        allZones = zones;
        rebuildZoneDropdowns(zones);
      });
  }


  // 5. Handle Apply Range button
  applyRangeButton.addEventListener("click", () => {
    const startZone = startSelect.value;
    const endZone = endSelect.value;

    if (!startZone || !endZone) {
      alert("Please select both start and end segments.");
      return;
    }

    localStorage.setItem("segmentStart", startZone);
    localStorage.setItem("segmentEnd", endZone);
    localStorage.removeItem("clickedZone");


    currentZone = null;
    currentStartZone = startZone;
    currentEndZone = endZone;

    fetch("/segment_data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        start_zone: startZone, 
        end_zone: endZone,
        lap: document.getElementById("lap-selector")?.value || null
    }),
    })
      .then(res => res.json())
      .then(data => {
        updatePlotsAndMap(data);

        const dropdown = document.getElementById("column-dropdown");
        const selectedKeyword = dropdown.value;

        if (selectedKeyword && selectedKeyword !== "Choose column...") {
          const queryParams = new URLSearchParams({
            keyword: selectedKeyword,
            zone: currentZone
          });
          const lap = document.getElementById("lap-selector")?.value;
          if (lap) queryParams.set("lap", lap);

          fetch(`/get_column_data?${queryParams.toString()}`)
            .then(res => res.json())
            .then(data => {
              availablePlotColumns[selectedKeyword] = Object.keys(data.cols || {});
              plotCurrentColumn();
            });
        }
      });
  });


  // 6. Upload CSV files
  document.getElementById("upload-form").addEventListener("submit", function (e) {
    e.preventDefault();

    const files = document.getElementById("csv-files").files;
    if (!files.length) {
      alert("Please select at least one CSV file to upload.");
      return;
    }

    const formData = new FormData();
    for (let file of files) {
      formData.append("files", file);
    }

    fetch("/upload_csv", {
      method: "POST",
      body: formData
    })
      .then(res => res.json())
      .then(data => {
        showToast(data.message);

        if (data.skipped_files && data.skipped_files.length > 0) {
          const msg = "Skipped files (missing timestamp):\n" + data.skipped_files.join(", ");
          showToast(msg, 6000);

          const listElement = document.getElementById("uploaded-files-list");
          data.skipped_files.forEach(file => {
            const li = document.createElement("li");
            li.textContent = file + " (skipped)";
            li.className = "list-group-item text-muted";
            listElement.appendChild(li);
          });
        }

        // Reset state
        currentZone = currentStartZone = currentEndZone = null;
        startSelect.selectedIndex = 0;
        endSelect.selectedIndex = 0;
        updateApplyButtonState();

        document.getElementById("track-map").contentWindow.postMessage({
          type: "highlight_segment",
          points: null
        }, "*");

        fetch("/segment_data", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({})
        })
          .then(res => res.json())
          .then(data => updatePlotsAndMap(data));

        loadZoneDropdowns();

        // Update column dropdown
        fetch("/get_combined_columns")
          .then(res => res.json())
          .then(columns => {
            const dropdown = document.getElementById("column-dropdown");
            dropdown.innerHTML = '<option disabled selected>Select column...</option>';
            columns.forEach(col => {
              const option = document.createElement("option");
              option.value = col.value;
              option.textContent = col.label;
              dropdown.appendChild(option);
            });
            updateUploadedFilesList();
          });

        // Update Lap dropdown
        fetch("/get_available_laps")
          .then(res => res.json())
          .then(laps => {
            const select = document.getElementById("lap-selector");
            select.innerHTML = "";

            const defaultOpt = document.createElement("option");
            defaultOpt.value = "";
            defaultOpt.textContent = "All Laps (full data)";
            select.appendChild(defaultOpt);          

            if (!laps.length) {
              document.getElementById("lap-selector-wrapper").style.display = "none";
              return;
            }

            laps.filter(lap => lap !== -1 && lap !== "-1")
            .forEach(lap => {
              const opt = document.createElement("option");
              opt.value = lap;
              opt.textContent = "Lap " + lap;
              select.appendChild(opt);
            });

            document.getElementById("lap-selector-wrapper").style.display = "block";
          })
          .then(() => {
            // Only trigger plot after column dropdown is also populated
            const dropdown = document.getElementById("column-dropdown");
            const selected = Array.from(dropdown.selectedOptions).filter(opt => !opt.disabled);
            if (selected.length) {
              plotCurrentColumn();
            }
          });

        // Get Trajectory Information
        fetch("/get_uploaded_trajectory")
          .then(res => res.json())
          .then(data => {
            if (data.lat && data.lon && data.lat.length > 0) {
              const points = data.lat.map((lat, i) => [lat, data.lon[i]]);
              document.getElementById("track-map").contentWindow.postMessage({
                type: "draw_uploaded_trajectory",
                points
              }, "*");
            }
          });

      });
  });


  // 7. Plot selected columns
  document.getElementById("submit-column").addEventListener("click", () => {
    const dd = document.getElementById("column-dropdown");
    selectedColumnKeywords = Array.from(dd.selectedOptions).filter(opt => !opt.disabled).map(opt => opt.value);

    localStorage.setItem("selectedColumns", JSON.stringify(selectedColumnKeywords));

    if (!selectedColumnKeywords.length) {
      alert("Please select at least one column group.");
      return;
    }

    availablePlotColumns = {};
    const promises = selectedColumnKeywords.map(kw => {
      const queryParams = new URLSearchParams({ keyword: kw });
      const lap = document.getElementById("lap-selector")?.value;
      if (lap) queryParams.set("lap", lap);
      if (currentZone) {
        queryParams.set("zone", currentZone);
      } else if (currentStartZone && currentEndZone) {
        queryParams.set("start_zone", currentStartZone);
        queryParams.set("end_zone", currentEndZone);
      }

      return fetch(`/get_column_data?${queryParams.toString()}`)
        .then(res => res.json())
        .then(data => {
          availablePlotColumns[kw] = Object.keys(data.cols || {});
        });
    });

    Promise.all(promises).then(() => {
      buildPlotFilterModal();
      new bootstrap.Modal(document.getElementById('plotFilterModal')).show();
    });
  });


  function buildPlotFilterModal() {
    const container = document.getElementById("plot-checkboxes");
    container.innerHTML = "";

    Object.entries(availablePlotColumns).forEach(([group, columns]) => {
      const groupContainer = document.createElement("div");
      groupContainer.className = "plot-group";

      const groupHeader = document.createElement("strong");
      groupHeader.textContent = formatGroupName(group);
      groupHeader.style.display = "block";
      groupHeader.style.marginBottom = "4px";
      groupContainer.appendChild(groupHeader);

      columns.forEach(col => {
        const id = `col_${col.replace(/\W+/g, "_")}`;
        const isChecked = !showAllByDefault && userSelectedPlots.has(col);

        const wrapper = document.createElement("div");
        wrapper.className = "form-check";

        wrapper.innerHTML = `
          <input class="form-check-input plot-option" type="checkbox" id="${id}" value="${col}" ${isChecked ? "checked" : ""}>
          <label class="form-check-label" for="${id}">${col}</label>
        `;

        groupContainer.appendChild(wrapper);
      });

      container.appendChild(groupContainer);
    });

    document.getElementById("showAllDefault").checked = showAllByDefault;

    setTimeout(() => {
      document.querySelectorAll(".plot-option").forEach(chk => {
        chk.addEventListener("change", () => {
          if (chk.checked) {
            document.getElementById("showAllDefault").checked = false;
          }
        });
      });
    }, 0);
  }


  function formatGroupName(keyword) {
    return keyword.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
  }


  document.getElementById("applyPlotFilter").addEventListener("click", () => {
    showAllByDefault = document.getElementById("showAllDefault").checked;
    userSelectedPlots = new Set();

    if (!showAllByDefault) {
      const checked = document.querySelectorAll("#plot-checkboxes input:checked");
      checked.forEach(chk => userSelectedPlots.add(chk.value));
    }

    localStorage.setItem("showAllByDefault", JSON.stringify(showAllByDefault));
    localStorage.setItem("userSelectedPlots", JSON.stringify([...userSelectedPlots]));  

    bootstrap.Modal.getInstance(document.getElementById("plotFilterModal")).hide();
    plotCurrentColumn();
  });


  // 8. Clear zone selections and reset map
  document.getElementById("clear-selection").addEventListener("click", function () {
    currentZone = currentStartZone = currentEndZone = null;
    startSelect.selectedIndex = 0;
    endSelect.selectedIndex = 0;
    updateApplyButtonState();

    document.getElementById("segment-title").innerText = "Zone: full";

    document.getElementById("track-map").contentWindow.postMessage({
      type: "highlight_segment",
      points: null
    }, "*");

    fetch("/segment_data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({})
    })
      .then(res => res.json())
      .then(data => {
        updatePlotsAndMap(data);
        plotCurrentColumn();
      });

    const dropdown = document.getElementById("column-dropdown");
    const selectedKeyword = dropdown.value;

    if (selectedKeyword && selectedKeyword !== "Choose column...") {
      plotCurrentColumn();
    }
  });


  // 9. Show toast message
  function showToast(message, duration = 3000) {
    const toast = document.getElementById("toast");
    toast.textContent = message;
    toast.style.visibility = "visible";

    setTimeout(() => {
      toast.style.visibility = "hidden";
    }, duration);
  }


  // 10. Plot currently selected column
  function plotCurrentColumn() {
    const dropdown = document.getElementById("column-dropdown");
    const selectedOptions = Array.from(dropdown.selectedOptions).filter(opt => !opt.disabled);
    const chartContainer = document.getElementById("chart-container");
    chartContainer.innerHTML = "";

    if (selectedOptions.length === 0) {
      alert("Please select one or more column groups.");
      return;
    }

    const lapSelector = document.getElementById("lap-selector");
    const lap = lapSelector?.value;

    const isGGIncluded = selectedOptions.some(opt => opt.value === "gg_plot");
    const nonGGOptions = selectedOptions.filter(opt => opt.value !== "gg_plot");

    const renderGGPlot = () => {
      if (!isGGIncluded) return Promise.resolve();

      const queryParams = new URLSearchParams({ keyword: "gg_plot" });
      if (lap) queryParams.set("lap", lap);
      if (currentZone) {
        queryParams.set("zone", currentZone);
      } else if (currentStartZone && currentEndZone) {
        queryParams.set("start_zone", currentStartZone);
        queryParams.set("end_zone", currentEndZone);
      }

      return fetch(`/get_column_data?${queryParams.toString()}`)
        .then(res => res.json())
        .then(ggData => {
          if (!ggData.x || !ggData.y) throw new Error("Invalid G-G plot data");

          const ggDiv = document.createElement("div");
          ggDiv.style.marginBottom = "30px";
          ggDiv.style.height = "400px";
          chartContainer.appendChild(ggDiv);

          Plotly.newPlot(ggDiv, [{
            x: ggData.x,
            y: ggData.y,
            mode: "markers",
            type: "scatter",
            name: "G-G Plot",
            marker: {
              size: 6,
              color: "rgba(100, 100, 255, 0.6)",
              line: { width: 1 }
            },
            hovertemplate: "Lat Acc: %{x:.2f} g<br>Long Acc: %{y:.2f} g<extra></extra>"
          }], {
            margin: { t: 20 },
            xaxis: {
              title: "Lateral Acceleration (g)",
              zeroline: true,
              zerolinewidth: 1,
              zerolinecolor: "#aaa",
              tickfont: { size: 11 },
              titlefont: { size: 11 }
            },
            yaxis: {
              title: "Longitudinal Acceleration (g)",
              zeroline: true,
              zerolinewidth: 1,
              zerolinecolor: "#aaa",
              tickfont: { size: 11 },
              titlefont: { size: 11 }
            },
            hovermode: "closest"
          });

          ggDiv.on("plotly_hover", ev => {
            const idx = ev.points[0].pointIndex;
            if (ggData.lat?.[idx] && ggData.lon?.[idx]) {
              document.getElementById("track-map").contentWindow.postMessage({
                type: "highlight_point",
                lat: ggData.lat[idx],
                lon: ggData.lon[idx]
              }, "*");
            }
          });
        });
    };

    const plots = [];
    let sharedX = null;
    let latLonRef = null;

    const plotPromises = nonGGOptions.map((option, idx) => {
      const kw = option.value;
      const text = option.textContent;
      const queryParams = new URLSearchParams({ keyword: kw });

      if (lap) queryParams.set("lap", lap);
      if (currentZone) {
        queryParams.set("zone", currentZone);
      } else if (currentStartZone && currentEndZone) {
        queryParams.set("start_zone", currentStartZone);
        queryParams.set("end_zone", currentEndZone);
      }

      return fetch(`/get_column_data?${queryParams.toString()}`)
        .then(res => res.json())
        .then(data => {
          if (data.error) throw new Error(data.error);

          if (!sharedX) {
            sharedX = data.x;
            latLonRef = { lat: data.lat, lon: data.lon };
          }

          const traces = Object.entries(data.cols)
            .filter(([name]) => showAllByDefault || userSelectedPlots.has(name))
            .map(([name, y]) => ({
              x: data.x,
              y,
              name,
              xaxis: "x",
              type: "scatter",
              mode: "lines",
              hovertemplate: `${name}: %{y}<extra></extra>`
            }));

          const yaxisName = `yaxis${idx === 0 ? '' : idx + 1}`;
          const yaxisShort = yaxisName.replace('yaxis', 'y');

          plots.push({
            traces: traces.map(t => ({ ...t, yaxis: yaxisShort })),
            axisName: yaxisName,
            title: text,
            domain: [
              1 - (idx + 1) / nonGGOptions.length,
              1 - idx / nonGGOptions.length
            ]
          });
        });
    });

    // Always render G-G plot first, then others
    renderGGPlot()
      .then(() => Promise.all(plotPromises))
      .then(() => {
        if (plots.length === 0) return;

        const allTraces = plots.flatMap(p => p.traces);
        const layout = {
          margin: { t: 20 },
          hovermode: "x unified",
          xaxis: { title: "Cumulative Distance (m)", domain: [0, 1] }
        };

        plots.forEach((p, idx) => {
          layout[p.axisName] = {
            title: p.title,
            anchor: "x",
            domain: p.domain,
            titlefont: { size: 11 },
            tickfont: { size: 11 }
          };
        });

        const chartDiv = document.createElement("div");
        chartDiv.style.width = "100%";
        chartDiv.style.height = `${plots.length * 200}px`;
        chartContainer.appendChild(chartDiv);

        Plotly.newPlot(chartDiv, allTraces, layout).then(() => {
          chartDiv.on("plotly_hover", ev => {
            const idx = ev.points[0].pointIndex;
            const lat = latLonRef.lat?.[idx];
            const lon = latLonRef.lon?.[idx];
            if (lat !== undefined && lon !== undefined) {
              document.getElementById("track-map").contentWindow.postMessage({
                type: "highlight_point", lat, lon
              }, "*");
            }
          });
        });
      })
      .catch(err => alert("Error while plotting: " + err.message));
  }


  // 11. Reset uploaded files
  document.getElementById("reset-uploads").addEventListener("click", () => {
    if (confirm("Are you sure you want to clear all uploaded files?")) {
      fetch("/reset_uploads", { method: "POST" })
        .then(res => res.json())
        .then(data => {
          alert(data.message);
          window.location.reload(); // Reload page to reset state
        });
    }
  });


  // 12. Update list of uploaded files
  function updateUploadedFilesList() {
    fetch("/get_uploaded_files")
      .then(response => response.json())
      .then(files => {
        const listElement = document.getElementById("uploaded-files-list");
        listElement.innerHTML = "";

        if (files.length === 0) {
          const li = document.createElement("li");
          li.textContent = "No files uploaded yet.";
          li.className = "list-group-item text-muted";
          listElement.appendChild(li);
          return;
        }

        files.forEach(file => {
          const li = document.createElement("li");
          li.textContent = file;
          li.className = "list-group-item";
          listElement.appendChild(li);
        });
      })
      .catch(error => console.error("Error fetching uploaded files:", error));
  }


  // 13. Handle Lap Selector Change
  document.getElementById("lap-selector").addEventListener("change", () => {
    const lap = document.getElementById("lap-selector").value;
    localStorage.setItem("selectedLap", lap);
    plotCurrentColumn();

    if (!lap || lap === "") {
      // No lap selected: show full uploaded trajectory
      fetch("/get_uploaded_trajectory")
        .then(res => res.json())
        .then(data => {
          if (data.lat && data.lon && data.lat.length > 0) {
            const points = data.lat.map((lat, i) => [lat, data.lon[i]]);
            document.getElementById("track-map").contentWindow.postMessage({
              type: "draw_uploaded_trajectory",
              points
            }, "*");
          }
        });
    } else {
      // Specific lap selected: show only lap trajectory
      const queryParams = new URLSearchParams({ lap });

      fetch(`/get_column_data?${queryParams.toString()}`)
        .then(res => res.json())
        .then(data => {
          if (data.lat && data.lon && data.lat.length > 0) {
            const points = data.lat.map((lat, i) => [lat, data.lon[i]]);
            document.getElementById("track-map").contentWindow.postMessage({
              type: "draw_uploaded_trajectory",
              points
            }, "*");
          }
        });
    }
  });
