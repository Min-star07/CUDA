#include <iostream>
#include <string>

// Define a custom struct
struct Person {
    std::string name;  // Member variable for the person's name
    int age;           // Member variable for the person's age

    // Constructor to initialize the struct (optional)
    Person(const std::string& name, int age) : name(name), age(age) {}

    // Optional: A member function to print the person's details
    void printDetails() const {
        std::cout << "Name: " << name << ", Age: " << age << std::endl;
    }
};

int main() {
    // Declare and initialize an instance of the struct
    Person person1("Alice", 25);

    // Access and modify members of the struct
    std::cout << person1.name << " is " << person1.age << " years old.\n";

    // Modify the age
    person1.age = 26;

    // Call a member function
    person1.printDetails();

    return 0;
}
