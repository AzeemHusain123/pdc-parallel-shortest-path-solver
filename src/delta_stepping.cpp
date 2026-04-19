// delta_stepping.cpp
// Compile: g++ delta_stepping.cpp -fopenmp -O2 -std=c++17 -o delta_stepping
// Usage: ./delta_stepping graph.txt delta num_threads
// graph format: n m  (first line) then m lines "u v w" (0-indexed, w >= 0 integers)

#include <bits/stdc++.h>
#include <omp.h>
#include <atomic>
using namespace std;
using ll = long long;
const ll INF = (1LL<<60);

void load_graph(const string &fn, vector<vector<pair<int,int>>> &g, int &n, long long &m) {
    ifstream in(fn);
    if (!in) { cerr << "Cannot open " << fn << "\n"; exit(1); }
    in >> n >> m;
    g.assign(n, {});
    int u,v,w;
    while (in >> u >> v >> w) g[u].push_back({v,w});
    in.close();
}

// Delta-stepping parallel SSSP
// delta: bucket width (positive integer). num_threads: OMP threads.
void delta_stepping(const vector<vector<pair<int,int>>> &g, int src, ll delta,
                    vector<ll> &dist_out, int num_threads) {

    int n = g.size();
    vector<atomic<ll>> dist(n);
    for (int i=0;i<n;i++) dist[i].store(INF);

    dist[src].store(0);

    // Light edges: weight <= delta ; Heavy: > delta
    vector<vector<pair<int,int>>> light(n), heavy(n);
    for (int u=0; u<n; ++u) {
        for (auto &e: g[u]) {
            if (e.second <= delta) light[u].push_back(e);
            else heavy[u].push_back(e);
        }
    }

    // buckets: map bucket_id -> vector of vertices
    // we store only non-empty buckets in an ordered set to find current min bucket
    using bucket_id_t = long long;
    unordered_map<bucket_id_t, vector<int>> buckets;
    buckets.reserve(1<<16);

    auto add_to_bucket = [&](int v, ll d) {
        bucket_id_t id = d / delta;
        auto &vec = buckets[id];
        vec.push_back(v);
    };

    // Initially put source in bucket 0
    add_to_bucket(src, 0);

    omp_set_num_threads(num_threads);

    // helper to relax a list of vertices' light edges in parallel, returns list of newly settled vertices
    auto relax_light = [&](vector<int> &B) {
        vector<int> newly;
        // parallel iterate vertices in B
        #pragma omp parallel
        {
            vector<int> local_new;
            #pragma omp for schedule(dynamic, 64)
            for (int i = 0; i < (int)B.size(); ++i) {
                int u = B[i];
                ll du = dist[u].load();
                for (auto &e: light[u]) {
                    int v = e.first; ll w = e.second;
                    ll nd = du + w;
                    ll cur = dist[v].load();
                    while (nd < cur) {
                        if (dist[v].compare_exchange_strong(cur, nd)) {
                            local_new.push_back(v);
                            break;
                        }
                        // cur updated; loop re-check
                    }
                }
            }
            // merge local_new into newly
            #pragma omp critical
            for (int x : local_new) newly.push_back(x);
        }
        return newly;
    };

    // helper to relax heavy edges (serial or parallel) from a set R (vertices)
    auto relax_heavy = [&](vector<int> &R) {
        // Parallel relax heavy edges, place into appropriate buckets
        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < (int)R.size(); ++i) {
            int u = R[i];
            ll du = dist[u].load();
            for (auto &e: heavy[u]) {
                int v = e.first; ll w = e.second;
                ll nd = du + w;
                ll cur = dist[v].load();
                while (nd < cur) {
                    if (dist[v].compare_exchange_strong(cur, nd)) {
                        // compute bucket id and push (need thread-safe push)
                        bucket_id_t id = nd / delta;
                        #pragma omp critical
                        buckets[id].push_back(v);
                        break;
                    }
                    // cur updated; loop re-check
                }
            }
        }
    };

    // main loop: process buckets in increasing id order
    while (!buckets.empty()) {
        // find smallest non-empty bucket id
        bucket_id_t cur_id = LLONG_MAX;
        for (auto &p : buckets) if (!p.second.empty()) cur_id = min(cur_id, p.first);
        if (cur_id == LLONG_MAX) break; // no work

        // extract B = buckets[cur_id], then clear it
        vector<int> B;
        {
            B.swap(buckets[cur_id]);
            buckets.erase(cur_id);
        }

        // R will collect vertices whose light-relax produced new distances within same bucket
        vector<int> R = B;

        // while R non-empty: relax light edges from R, adding newly settled vertices to R
        while (!R.empty()) {
            vector<int> newly = relax_light(R);
            R.swap(newly);
            // newly will be emptied and loop repeats
        }

        // After finishing light relaxations for bucket cur_id, relax heavy edges for all vertices in B
        relax_heavy(B);
    }

    // copy out
    dist_out.assign(n, INF);
    for (int i=0;i<n;i++) dist_out[i] = dist[i].load();
}

int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "Usage: ./delta_stepping graph.txt delta num_threads\n";
        return 1;
    }
    string fn = argv[1];
    ll delta = atoll(argv[2]);
    int threads = stoi(argv[3]);

    vector<vector<pair<int,int>>> g;
    int n; long long m;
    load_graph(fn, g, n, m);
    cerr << "Loaded n="<<n<<" m="<<m<<"\n";
    int src, dest;
    cout << "Enter source (0-" << n-1 << "): "; cin >> src;
    cout << "Enter dest   (0-" << n-1 << "): "; cin >> dest;

    vector<ll> dist;
    double t0 = omp_get_wtime();
    delta_stepping(g, src, delta, dist, threads);
    double t1 = omp_get_wtime();
    cout << "Delta-stepping time: " << (t1-t0) << " sec\n";
    cout << "Distance to " << dest << " = " << dist[dest] << "\n";
    return 0;
}
