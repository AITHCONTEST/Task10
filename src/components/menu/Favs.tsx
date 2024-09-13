import { observer } from "mobx-react-lite";
import favsStore from "../../store/favsStore";
import styles from "./../styles/menu.module.scss"
import textService from "../../service/textService";
interface FavsProps {
    tap: () => void;
}
const Favs = observer(({tap}: FavsProps) => {
   

    return(
        <div className={styles.favs} >
            <div>Закрепленные переводы:</div>
            <br />
          {
            favsStore.items.map((item)=>(
                <div 
                    className={styles.favs__item}
                    onClick={()=>{textService.setInput(item); favsStore.add(item); tap()}}
                >
                    {item}
                </div>
            ))
          }
        </div>
    );
});

export default Favs;