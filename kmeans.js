// JavaScript for KMeans Clustering and Visualization

let data = [];
let centers = [];
let assignments = [];
let converged = false;
let initMethod = 'random';
let k = 3; // Default number of clusters
const width = 600;
const height = 600;
const svg = d3.select("#scatterplot");
const colors = ["blue", "green", "red", "orange", "purple", "cyan", "magenta", "yellow", "black", "brown"];
let manualCentroidMode = false; // For manual centroid selection

// Generate random dataset
function generateData() {
    data = [];
    for (let i = 0; i < 300; i++) {
        data.push({
            x: Math.random() * width,
            y: Math.random() * height,
            cluster: -1
        });
    }
    centers = [];
    assignments = Array(data.length).fill(-1);
    converged = false;
    draw();
}

// Update k based on user input
function updateK() {
    const kInput = document.getElementById("k-value").value;
    k = Math.max(1, Math.min(10, parseInt(kInput))); // Constrain k between 1 and 10
}

// Random initialization of centroids
function randomInit() {
    centers = [];
    const selectedIndices = new Set();
    while (centers.length < k) {
        let idx = Math.floor(Math.random() * data.length);
        if (!selectedIndices.has(idx)) {
            selectedIndices.add(idx);
            centers.push({ ...data[idx] });
        }
    }
}

// Farthest first initialization of centroids
function farthestFirstInit() {
    centers = [];
    const selected = new Set();
    let firstIdx = Math.floor(Math.random() * data.length);
    centers.push({ ...data[firstIdx] });
    selected.add(firstIdx);

    for (let i = 1; i < k; i++) {
        let maxDistance = -1;
        let farthestPoint = null;
        for (let j = 0; j < data.length; j++) {
            if (selected.has(j)) continue;
            let minDist = Math.min(...centers.map(center => distance(data[j], center)));
            if (minDist > maxDistance) {
                maxDistance = minDist;
                farthestPoint = { ...data[j] };
            }
        }
        centers.push(farthestPoint);
    }
}

// KMeans++ initialization
function kmeansPlusPlusInit() {
    centers = [];
    let firstIdx = Math.floor(Math.random() * data.length);
    centers.push({ ...data[firstIdx] });

    for (let i = 1; i < k; i++) {
        let distances = data.map(point => Math.min(...centers.map(center => distance(point, center))));
        let totalDistance = d3.sum(distances);
        let probabilities = distances.map(d => d / totalDistance);
        let cumProbs = d3.cumsum(probabilities);
        let randomVal = Math.random();

        for (let j = 0; j < cumProbs.length; j++) {
            if (randomVal < cumProbs[j]) {
                centers.push({ ...data[j] });
                break;
            }
        }
    }
}

// Manual centroid selection
function manualInit() {
    manualCentroidMode = true;
    centers = [];
    svg.on("click", function(event) {
        if (centers.length < k) {
            const [x, y] = d3.pointer(event);
            centers.push({ x, y });
            draw();
            if (centers.length === k) {
                manualCentroidMode = false;
                svg.on("click", null); // Disable clicking after centroids are selected
            }
        }
    });
}

// Assign each point to the nearest centroid
function assignClusters() {
    data.forEach((point, idx) => {
        let minDist = Infinity;
        let bestCenter = -1;
        centers.forEach((center, centerIdx) => {
            let dist = distance(point, center);
            if (dist < minDist) {
                minDist = dist;
                bestCenter = centerIdx;
            }
        });
        assignments[idx] = bestCenter;
    });
}

// Recompute centroids based on current assignments
function recomputeCenters() {
    let newCenters = Array(k).fill(null).map(() => ({ x: 0, y: 0, count: 0 }));

    data.forEach((point, idx) => {
        let clusterIdx = assignments[idx];
        newCenters[clusterIdx].x += point.x;
        newCenters[clusterIdx].y += point.y;
        newCenters[clusterIdx].count += 1;
    });

    centers = newCenters.map(center => ({
        x: center.x / (center.count || 1),
        y: center.y / (center.count || 1)
    }));
}

// Check for convergence (centroids no longer change)
function checkConvergence(oldCenters) {
    return oldCenters.every((center, idx) => {
        return distance(center, centers[idx]) < 1e-5;
    });
}

// Perform one step of the KMeans algorithm
function step() {
    if (converged) return;
    let oldCenters = centers.map(center => ({ ...center }));
    assignClusters();
    recomputeCenters();
    converged = checkConvergence(oldCenters);
    draw();
}

// Go straight to convergence
function converge() {
    while (!converged) {
        step();
    }
}

// Initialize centers (without clustering yet)
function initializeCenters() {
    updateK();
    const method = document.getElementById("init-method").value;
    if (method === "random") randomInit();
    if (method === "farthest") farthestFirstInit();
    if (method === "kmeans++") kmeansPlusPlusInit();
    if (method === "manual") manualInit();
    draw();
}

// Reset the clustering process
function reset() {
    generateData();
    assignments = Array(data.length).fill(-1);
    centers = [];
    converged = false;
    draw();
}

// Calculate distance between two points
function distance(p1, p2) {
    return Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
}

// Draw the data points and centroids
function draw() {
    svg.selectAll("*").remove();

    // Draw data points
    svg.selectAll(".point")
        .data(data)
        .enter()
        .append("circle")
        .attr("cx", d => d.x)
        .attr("cy", d => d.y)
        .attr("r", 4)
        .attr("fill", (d, i) => colors[assignments[i]] || "gray");

    // Draw centroids
    svg.selectAll(".centroid")
        .data(centers)
        .enter()
        .append("circle")
        .attr("cx", d => d.x)
        .attr("cy", d => d.y)
        .attr("r", 10)
        .attr("fill", "black");
}

// Event Listeners
document.getElementById("new-dataset").addEventListener("click", generateData);
document.getElementById("step").addEventListener("click", step);
document.getElementById("converge").addEventListener("click", converge);
document.getElementById("reset").addEventListener("click", reset);
document.getElementById("initialize-centers").addEventListener("click", initializeCenters); // Button for initializing centers
document.getElementById("manual-init").addEventListener("click", manualInit); // Button for manual centroid selection

// Initial setup
generateData();
reset();
