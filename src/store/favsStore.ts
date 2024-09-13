import { makeAutoObservable } from "mobx";
import IFavsStore from "../model/favsStore";


class FavsStore implements IFavsStore {
  public items: string[] = ["Привет"];
  private static instance: FavsStore;
  private storageKey = "favsItems";

  private constructor() {
    this.loadFromLocalStorage();
    makeAutoObservable(this);
  }

  public static getInstance(): FavsStore {
    if (!FavsStore.instance) {
      FavsStore.instance = new FavsStore();
    }
    return FavsStore.instance;
  }


  public exists(item: string): boolean {
    return this.items.includes(item);
  }


  public add(item: string): void {
    const index = this.items.indexOf(item);
    if (index !== -1) {
      this.items.splice(index, 1);
    }
    this.items.unshift(item);
    this.saveToLocalStorage();
  }

  public remove(item: string): void {
    const index = this.items.indexOf(item);
    if (index !== -1) {
      this.items.splice(index, 1);
    }
    this.saveToLocalStorage();
  }

  private saveToLocalStorage(): void {
    localStorage.setItem(this.storageKey, JSON.stringify(this.items));
  }

  // Load the array from localStorage
  private loadFromLocalStorage(): void {
    const storedItems = localStorage.getItem(this.storageKey);
    if (storedItems) {
      this.items = JSON.parse(storedItems);
    }
  }
}

export default FavsStore.getInstance();


    // private saveToLocalStorage(): void {
    //     localStorage.setItem('fromLang', this.fromLang);
    //     localStorage.setItem('toLang', this.toLang);
    // }

    // private loadFromLocalStorage(): boolean {
    //     const fromLang = localStorage.getItem('fromLang');
    //     if (fromLang) {
    //         this.fromLang = fromLang;
    //     }
    //     const toLang = localStorage.getItem('toLang');
    //     if (toLang) {
    //         this.toLang = toLang;
    //     }
    //     return !!fromLang && !!toLang;
    // }


