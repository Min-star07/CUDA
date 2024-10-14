#include <iostream>

struct mylist {
    int key;          // The value stored in the node
    mylist *next;     // Pointer to the next node

    // Constructor for ease of node creation
    mylist(int value) : key(value), next(nullptr) {}
};

int main() {
    // Create the head of the linked list
    mylist *list_elem = new mylist(0); // Start with the first element (key = 0)
    mylist *current = list_elem;        // Pointer to track the current node

    // Create and link 3 more nodes (for keys 1, 2, and 3)
    for (int i = 1; i < 4; i++) {
        // Allocate memory for a new node
        
        current->next = new mylist(0); // Allocate memory for the next node
        // Now assign the key value
        current->next->key = i;         // Set the key value
        
        // current->next = new mylist(i); // Create a new node with key = i
        current = current->next;        // Move the current pointer to the new node
    }

    // Pointer to traverse the list and print keys
    mylist *ele = list_elem;
    while (ele != nullptr) {
        std::cout << ele->key << std::endl; // Print the key of the current node
        ele = ele->next; // Move to the next node
    }

    // Cleanup: Free allocated memory
    current = list_elem; // Reset current to the head
    mylist *next_node;
    while (current != nullptr) {
        next_node = current->next; // Save the next node
        delete current; // Delete the current node
        current = next_node; // Move to the next node
    }

    return 0;
}
