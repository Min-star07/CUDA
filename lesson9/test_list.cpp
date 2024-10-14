#include<iostream>
#include<list>
int main(){
    std::list<int> myList = {1, 2,  4, 5};
    auto it = myList.begin();
    std::advance(it, 2);
    myList.insert(it, 3);
    myList.reverse();
    for (int num : myList)
    {
        std::cout << num << "\t" << std::endl;
    }
    return 0;
}