interface IFavsStore {
    items: string[];
    exists(item: string): boolean;
    add(item: string): void;
    remove(item: string): void;
}
  
export default IFavsStore;