import { makeAutoObservable } from "mobx";
import IFavsStore from "../model/favsStore";
import FavsItem from "../model/favsItem";


class FavsStore implements IFavsStore {
  public items: FavsItem[] = [{ lang: "rus", text: "Привет" }];
  private static instance: FavsStore;
  private storageKey = "FavsItems";

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

  public exists(item: FavsItem): boolean {
    return this.items.some(
      (fav) => fav.lang === item.lang && fav.text === item.text
    );
  }

  public add(item: FavsItem): void {
    const index = this.items.findIndex(
      (fav) => fav.lang === item.lang && fav.text === item.text
    );
    if (index !== -1) {
      this.items.splice(index, 1); // Если элемент уже существует, удаляем его
    }
    this.items.unshift(item); // Добавляем элемент в начало
    this.saveToLocalStorage();
  }

  public remove(item: FavsItem): void {
    const index = this.items.findIndex(
      (fav) => fav.lang === item.lang && fav.text === item.text
    );
    if (index !== -1) {
      this.items.splice(index, 1); // Удаляем элемент
    }
    this.saveToLocalStorage();
  }

  private saveToLocalStorage(): void {
    localStorage.setItem(this.storageKey, JSON.stringify(this.items));
  }

  private loadFromLocalStorage(): void {
    const storedItems = localStorage.getItem(this.storageKey);
    if (storedItems) {
      this.items = JSON.parse(storedItems);
    }
  }
}

export default FavsStore.getInstance();
