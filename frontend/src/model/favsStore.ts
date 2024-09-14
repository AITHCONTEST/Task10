import FavsItem from "./favsItem";

interface IFavsStore {
    exists(item: FavsItem): boolean;
    add(item: FavsItem): void;
    remove(item: FavsItem): void;
}
  
export default IFavsStore;
