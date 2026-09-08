#include <bits/stdc++.h>
#include<iostream>
using namespace std;

int main() {
    int t;
    cin >> t;

    while (t--) {
        long long a, b, x;
        cin >> a >> b >> x;

        long long A = a, B = b;

        unordered_map<long long,long long> m_a;
        unordered_map<long long,long long> m_b;

        long long cost_a = 0;
        while (true) {
            m_a[a] = cost_a;

            if (a == 0) break;

            a /= x;
            cost_a++;
        }

        long long cost_b = 0;
        while (true) {
            m_b[b] = cost_b;

            if (b == 0) break;

            b /= x;
            cost_b++;
        }

        long long ans = abs(A - B);

        for (auto &[va, ca] : m_a) {
            for (auto &[vb, cb] : m_b) {
                ans = min(ans,
                          ca + cb + abs(va - vb));
            }
        }

        cout << ans << '\n';
    }

    return 0;
}