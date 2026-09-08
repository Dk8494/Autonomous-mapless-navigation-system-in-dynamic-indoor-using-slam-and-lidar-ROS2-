#include <bits/stdc++.h>
#include <iostream>
using namespace std;

int main()
{
    int t;
    cin >> t;
    while (t--)
    {
        int n;
        cin >> n;
        long long mini=INT_MAX;
        long long maxi=INT_MIN;
        for (int i = 0; i < n; i++)
        {
            long long x;
            cin>>x;
            maxi=max(maxi,x);
            mini=min(mini,x);
        }
        cout<<maxi-mini+1<<endl;
    }
    return 0;
}