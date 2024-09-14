import { observer } from "mobx-react-lite";
import favsStore from "../../store/favsStore";
import styles from "./../styles/menu.module.scss"
import textService from "../../service/textService";
import langStore from "../../store/langStore";
import langService from "../../service/langService";

interface FavsProps {
    tap: () => void;
}
const Favs = observer(({tap}: FavsProps) => {
   

    return(
        <div className={styles.favs} >
            <div><b>Закрепленные переводы:</b></div>
            <br />
          {
            favsStore.items.map((item)=>(
                <div 
                    className={styles.favs__item}
                    onClick={()=>{
                        textService.setInput(item.text); 
                        favsStore.add(item);
                        const fromLang =langStore.getFromlang();
                        if(item.lang != fromLang){
                            langService.toggleLang();
                        } 
                        tap();
                    }}
                >
                    {item.text}
                </div>
            ))
          }
        </div>
    );
});

export default Favs;