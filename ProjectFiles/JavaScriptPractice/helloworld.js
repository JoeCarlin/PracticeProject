// Variables
let name = "John";
const age = 30;
var isStudent = true;

// Functions
function greet(person) {
    return `Hello, ${person}!`;
}

// Arrow Functions
const add = (a, b) => a + b;

// Objects
const person = {
    firstName: "Jane",
    lastName: "Doe",
    age: 25,
    greet() {
        return `Hello, my name is ${this.firstName} ${this.lastName}.`;
    }
};

// Arrays
const numbers = [1, 2, 3, 4, 5];

// Array Methods
numbers.forEach(num => console.log(num));
const doubled = numbers.map(num => num * 2);

// Conditionals
if (age > 18) {
    console.log("Adult");
} else {
    console.log("Minor");
}

// Loops
for (let i = 0; i < numbers.length; i++) {
    console.log(numbers[i]);
}

for (const num of numbers) {
    console.log(num);
}

// Classes
class Animal {
    constructor(name, species) {
        this.name = name;
        this.species = species;
    }

    speak() {
        console.log(`${this.name} makes a noise.`);
    }
}

class Dog extends Animal {
    constructor(name, breed) {
        super(name, "Dog");
        this.breed = breed;
    }

    speak() {
        console.log(`${this.name} barks.`);
    }
}

// Promises
const fetchData = () => {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            resolve("Data fetched");
        }, 2000);
    });
};

fetchData().then(data => console.log(data)).catch(err => console.error(err));

// Async/Await
const fetchDataAsync = async () => {
    try {
        const data = await fetchData();
        console.log(data);
    } catch (err) {
        console.error(err);
    }
};

fetchDataAsync();