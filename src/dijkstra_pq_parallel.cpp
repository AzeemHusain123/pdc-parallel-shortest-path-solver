// dijkstra_pq_parallel.cpp
// Compile: g++ dijkstra_pq_parallel.cpp -fopenmp -O2 -std=c++17 -o dijkstra_pq_parallel
// Usage: ./dijkstra_pq_parallel graph.txt num_threads
// graph format: n m  (first line), then m lines "u v w" (0-indexed), directed edges.

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

// Sequential Dijkstra (for reference & path parent)
void dijkstra_seq_parent(const vector<vector<pair<int,int>>> &g, int src,
                         vector<ll> &dist, vector<int> &parent) {
    int n = g.size();
    dist.assign(n, INF);
    parent.assign(n, -1);
    using pli = pair<ll,int>;
    priority_queue<pli, vector<pli>, greater<pli>> pq;
    dist[src] = 0; pq.push({0, src});
    while(!pq.empty()) {
        auto [d,u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        for (auto &e: g[u]) {
            int v = e.first, w = e.second;
            ll nd = d + w;
            if (nd < dist[v]) { dist[v]=nd; parent[v]=u; pq.push({nd,v}); }
        }
    }
}

// Parallel Dijkstra: serial extract-min, parallel relaxations using atomic dist[]
void dijkstra_pq_parallel(const vector<vector<pair<int,int>>> &g, int src,
                          vector<ll> &dist_out, int num_threads) {
    int n = g.size();
    vector<atomic<ll>> dist_atomic(n);
    for (int i=0;i<n;i++) dist_atomic[i].store(INF);
    dist_atomic[src].store(0);

    using pli = pair<ll,int>;
    priority_queue<pli, vector<pli>, greater<pli>> pq;
    pq.push({0, src});
    mutex pq_mtx;

    omp_set_num_threads(num_threads);

    while (true) {
        pli top;
        bool have = false;
        {
            // serial extract-min
            lock_guard<mutex> lock(pq_mtx);
            while(!pq.empty()) {
                top = pq.top();
                if (top.first != dist_atomic[top.second].load()) { pq.pop(); continue; }
                pq.pop();
                have = true;
                break;
            }
        }
        if (!have) break;
        ll d = top.first; int u = top.second;

        auto &adj = g[u];
        int deg = (int)adj.size();

        // parallel relax neighbors
        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < deg; ++i) {
            int v = adj[i].first;
            int w = adj[i].second;
            ll nd = d + (ll)w;

            ll cur = dist_atomic[v].load();
            while (nd < cur) {
                // attempt CAS: if dist[v] == cur, set to nd
                if (dist_atomic[v].compare_exchange_strong(cur, nd)) {
                    // successful update, push into pq
                    lock_guard<mutex> lock(pq_mtx);
                    pq.push({nd, v});
                    break;
                }
                // cur updated by others; if still larger, loop again with new cur
            }
        }
    }

    // copy out
    dist_out.assign(n, INF);
    for (int i=0;i<n;i++) dist_out[i] = dist_atomic[i].load();
}

vector<int> reconstruct_path(const vector<int> &parent, int src, int dest) {
    vector<int> path;
    int cur = dest;
    while (cur != -1) {
        path.push_back(cur);
        if (cur == src) break;
        cur = parent[cur];
    }
    reverse(path.begin(), path.end());
    if (path.empty() || path[0] != src) path.clear();
    return path;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: ./dijkstra_pq_parallel graph.txt num_threads\n"; return 1;
    }
    string fn = argv[1];
    int threads = stoi(argv[2]);

    vector<vector<pair<int,int>>> g;
    int n; long long m;
    load_graph(fn, g, n, m);
    cerr << "Loaded n="<<n<<" m="<<m<<"\n";

    int src, dest;
    cout << "Enter source (0-" << n-1 << "): "; cin >> src;
    cout << "Enter dest   (0-" << n-1 << "): "; cin >> dest;

    vector<ll> dist_seq; vector<int> parent;
    double t0 = omp_get_wtime();
    dijkstra_seq_parent(g, src, dist_seq, parent);
    double t1 = omp_get_wtime();
    cout << "Seq time: " << (t1-t0) << " sec\n";

    vector<ll> dist_par;
    double p0 = omp_get_wtime();
    dijkstra_pq_parallel(g, src, dist_par, threads);
    double p1 = omp_get_wtime();
    cout << "Par time ("<<threads<<" threads): " << (p1-p0) << " sec\n";

    cout << "Distance seq["<<dest<<"] = " << dist_seq[dest] << "\n";
    vector<int> path = reconstruct_path(parent, src, dest);
    if (path.empty()) cout << "No path (seq parent)\n";
    else {
        cout << "Path (seq parent): ";
        for (size_t i=0;i<path.size();++i) { if (i) cout<<" -> "; cout<<path[i]; } cout<<"\n";
    }
    return 0;
}
