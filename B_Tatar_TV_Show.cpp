#include <bits/stdc++.h>
#include<iostream>
using namespace std;

int main() {
    int t;
    cin>>t;
    while(t--){
        int n,k;
        cin>>n>>k;
        string s;
        int ind;
        cin>>s;
        bool ispos=true;
        for(int i=0;i<k;i++){
            int cnt=0;
            for(int j=i;j<n;j+=k){
                if(s[j]=='1') cnt++;
            }
            if(cnt%2){
                ispos=false;
                break;
            }
        }
        if(ispos) cout<<"YES"<<endl;
        else cout<<"NO"<<endl;
    }
    return 0;
}