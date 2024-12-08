/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package model;

import java.util.ArrayList;

/**
 *
 * @author tawde
 */
public class PersonDirectory {
    
    private ArrayList<Person> personList;

    public PersonDirectory() {
        this.personList = new ArrayList<Person>();
    }

    public ArrayList<Person> getPersonList() {
        return personList;
    }

    public void setPersonList(ArrayList<Person> personList) {
        this.personList = personList;
    }
    
    public Person addPerson() {
        Person person = new Person();
        personList.add(person);
        return person;
    }
    
    public void deletePerson(Person person) {
        personList.remove(person);
    }
    
        public Person searchPerson(String searchInput) {
        for(Person person: personList) {
            if(person.getFirstName().equals(searchInput) || person.getLastName().equals(searchInput) || person.getWorkAddress().getStreetAddress().equals(searchInput) || person.getHomeAddress().getStreetAddress().equals(searchInput)) {
                return person;
            }
        }
        return null;
    }
    
    
}
